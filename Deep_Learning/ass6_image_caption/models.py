"""
Code base inspired by: https://gitcode.com/sgrvinod/a-pytorch-tutorial-to-image-captioning/overview and
"""

import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encoder
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        # Pretrained ImageNet ResNet-101
        # Remove linear and pool layers
        resnet = torchvision.models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune(fine_tune=True)

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: boolean
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class DecoderWithRNN(nn.Module):
    def __init__(self, cfg, encoder_dim=14 * 14 * 2048):
        """
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithRNN, self).__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = cfg['decoder_dim']
        self.embed_dim = cfg['embed_dim']
        self.vocab_size = cfg['vocab_size']
        self.dropout = cfg['dropout']
        self.device = cfg['device']

        self.embedding = nn.Embedding(cfg['vocab_size'], cfg['embed_dim']) #(size of dictionary of embedding, each embedding vector)
        self.decode_step = nn.LSTMCell(cfg['embed_dim'], cfg['decoder_dim'])
        self.fc = nn.Linear(cfg['decoder_dim'], cfg['vocab_size'], bias=True)

        self.init = nn.Linear(encoder_dim, cfg['decoder_dim'])
        self.bn = nn.BatchNorm1d(cfg['decoder_dim'], momentum=0.01)
        self.fc = nn.Linear(cfg['decoder_dim'], cfg['vocab_size'])
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # initialize some layers with the uniform distribution
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions(ground truth), a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths(from label info), a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_out = encoder_out.reshape(batch_size, -1)
        vocab_size = self.vocab_size

        # Sort input data by decreasing lengths;
        # these three iterators are originally ordered the same by the image order
        # therefore by changing one's order, the others will change accordingly
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        # basically a lookup table, input is a shape contains the indices to extract
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1(the goal decode length from read-in data)
        decode_lengths = (caption_lengths - 1).tolist()
        # Create tensors to hold word prediction scores
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(self.device)

        # Initialize LSTM state
        # the initial state is retrieved by passing the encoding result through a fc layer with normalization
        # then decode the normalized results
        init_input = self.bn(self.init(encoder_out))
        h, c = self.decode_step(init_input)  # (batch_size_t, decoder_dim)

        # for each time step
        for t in range(max(decode_lengths)):
            # retrieve the index for current timestep
            # remember that each image has different decode length so need to do add up
            i = sum([l > t for l in decode_lengths])
            preds, h, c = self.one_step(
                embeddings[:i, t, :], h[:i], c[:i])
            predictions[:i, t, :] = preds
        return predictions, encoded_captions, decode_lengths, sort_ind

    def one_step(self, embeddings, h, c):
        h, c = self.decode_step(embeddings, (h, c))
        # Compute the scores over the vocabulary
        preds = self.fc(self.dropout_layer(h))
        return preds, h, c


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()

        self.encoder_attention = nn.Linear(encoder_dim, attention_dim)
        self.decoder_attention = nn.Linear(decoder_dim, attention_dim)
        # attention layer
        self.att = nn.Linear(attention_dim, 1) # 1 -> in_features
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward pass.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        # "e = f_att(encoder_out, decoder_hidden)"
        # "alpha = softmax(e)"
        # "z = alpha * encoder_out"
        encode_attention = self.encoder_attention(encoder_out)  # (batch_size, num_pixels, attention_dim)
        decode_attention = self.decoder_attention(decoder_hidden)  # (batch_size, attention_dim)
        # (batch_size, num_pixels, attention_dim) -> (batch_size, num_pixels)
        e = self.att(self.relu(encode_attention + decode_attention.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(e)  # (batch_size, num_pixels)
        z = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        return z, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, cfg, encoder_dim=2048):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = cfg['decoder_dim']
        self.attention_dim = cfg['attention_dim']
        self.embed_dim = cfg['embed_dim']
        self.vocab_size = cfg['vocab_size']
        self.dropout = cfg['dropout']
        self.device = cfg['device']

        self.attention = Attention(self.encoder_dim, self.decoder_dim, self.attention_dim) # attention layer
        self.embedding = nn.Embedding(cfg['vocab_size'], cfg['embed_dim']) # same as RNN above -> lookup tabel
        self.decode_step = nn.LSTMCell(cfg['embed_dim'] + self.encoder_dim, cfg['decoder_dim']) # LSTM decoder
        self.init_h = nn.Linear(self.encoder_dim, cfg['decoder_dim']) # initialize LSTM -> h
        self.init_c = nn.Linear(self.encoder_dim, cfg['decoder_dim']) # initialize LSTM -> c
        self.beta = nn.Linear(self.decoder_dim, 1) # sigmoid gate
        self.fc = nn.Linear(cfg['decoder_dim'], cfg['vocab_size']) # FC layer for probability
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.bn = nn.BatchNorm1d(cfg['decoder_dim'], momentum=0.01)

        # initialize some layers with the uniform distribution
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths;
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(self.device)

        # Initialize LSTM state
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)

        # Teacher forcing is used.
        # At each time-step, decode by attention-weighting the encoder's output based
        for t in range(max(decode_lengths)):
            i = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:i], h[:i])
            gate = self.beta(h[:i])
            attention_weighted_encoding = gate * attention_weighted_encoding
            decoder_teacher_forcing = torch.cat([embeddings[:i, t, :], attention_weighted_encoding], dim=1)
            h, c = self.decode_step(decoder_teacher_forcing, (h[:i], c[:i]))
            preds = self.fc(self.dropout_layer(h))
            predictions[:i, t, :] = preds
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

    def one_step(self, embeddings, encoder_out, h, c):
        # this function can be used for test decode with beam search
        # return predicted scores over vocabs: preds
        # return attention weight: alpha
        # return hidden state and cell state: h, c
        # Your Code Here!
        attention_weighted_encoding, alpha = self.attention(encoder_out, h)
        gate = self.beta(h)
        attention_weighted_encoding = gate * attention_weighted_encoding
        decoder_input = torch.cat([embeddings, attention_weighted_encoding], dim=1)
        h, c = self.decode_step(decoder_input, (h, c))
        preds = self.fc(self.dropout_layer(h))
        return preds, alpha, h, c