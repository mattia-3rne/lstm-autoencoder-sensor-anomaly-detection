from utils import sigmoid, tanh, d_sigmoid, d_tanh
import numpy as np

class LSTMCell:
    def __init__(self, input_dim, hidden_dim):
        self.p = hidden_dim
        self.W = np.random.randn(4 * self.p, input_dim) * 0.01
        self.U = np.random.randn(4 * self.p, self.p) * 0.01
        self.b = np.zeros((4 * self.p, 1))

    def forward(self, x, h_prev, c_prev):
        z = self.W @ x + self.U @ h_prev + self.b
        
        f = sigmoid(z[0:self.p])
        i = sigmoid(z[self.p:2*self.p])
        g = tanh(z[2*self.p:3*self.p])
        o = sigmoid(z[3*self.p:4*self.p])
        
        c = f * c_prev + i * g
        h = o * tanh(c)
        
        cache = (x, h_prev, c_prev, f, i, g, o, c, z)
        return h, c, cache

    def backward(self, dh, dc, cache):
        x, h_prev, c_prev, f, i, g, o, c, z = cache
        
        do = dh * tanh(c)
        dc += dh * o * d_tanh(c)
        
        df = dc * c_prev
        di = dc * g
        dg = dc * i
        
        dz = np.vstack((
            df * d_sigmoid(z[0:self.p]),
            di * d_sigmoid(z[self.p:2*self.p]),
            dg * d_tanh(z[2*self.p:3*self.p]),
            do * d_sigmoid(z[3*self.p:4*self.p])
        ))
        
        dW = dz @ x.T
        dU = dz @ h_prev.T
        db = dz
        
        dx = self.W.T @ dz
        dh_prev = self.U.T @ dz
        dc_prev = f * dc
        
        return dx, dh_prev, dc_prev, dW, dU, db

class LSTMAutoencoder:
    def __init__(self, seq_len, p, q):
        self.T = seq_len
        self.p = p
        self.q = q
        
        self.encoder = LSTMCell(1, p)
        self.decoder = LSTMCell(p, q)
        
        self.W_R = np.random.randn(1, q) * 0.01
        self.b_R = np.zeros((1, 1))

    def forward(self, w):
        h_enc = np.zeros((self.p, 1))
        c_enc = np.zeros((self.p, 1))
        enc_caches = []
        
        for t in range(self.T):
            x_t = w[t].reshape(-1, 1)
            h_enc, c_enc, cache = self.encoder.forward(x_t, h_enc, c_enc)
            enc_caches.append(cache)
            
        z = h_enc
        c_dec = c_enc
        h_dec = np.zeros((self.q, 1))
        dec_caches = []
        r = np.zeros((self.T, 1))
        h_dec_seq = []
        
        for s in range(self.T):
            h_dec, c_dec, cache = self.decoder.forward(z, h_dec, c_dec)
            dec_caches.append(cache)
            h_dec_seq.append(h_dec)
            r[s] = self.W_R @ h_dec + self.b_R
            
        return r, enc_caches, dec_caches, h_dec_seq

    def update_params(self, grads, lr):
        dW_enc, dU_enc, db_enc, dW_dec, dU_dec, db_dec, dW_R, db_R = grads
        
        self.encoder.W -= lr * dW_enc
        self.encoder.U -= lr * dU_enc
        self.encoder.b -= lr * db_enc
        
        self.decoder.W -= lr * dW_dec
        self.decoder.U -= lr * dU_dec
        self.decoder.b -= lr * db_dec
        
        self.W_R -= lr * dW_R
        self.b_R -= lr * db_R

    def save_model(self, filepath, history=None):
        save_dict = {
            "W_enc": self.encoder.W, "U_enc": self.encoder.U, "b_enc": self.encoder.b,
            "W_dec": self.decoder.W, "U_dec": self.decoder.U, "b_dec": self.decoder.b,
            "W_R": self.W_R, "b_R": self.b_R
        }
        if history is not None:
            save_dict["loss_history"] = np.array(history)
            
        np.savez(filepath, **save_dict)

    def load_model(self, filepath):
        data = np.load(filepath)
        self.encoder.W = data['W_enc']
        self.encoder.U = data['U_enc']
        self.encoder.b = data['b_enc']
        self.decoder.W = data['W_dec']
        self.decoder.U = data['U_dec']
        self.decoder.b = data['b_dec']
        self.W_R = data['W_R']
        self.b_R = data['b_R']
        
        if 'loss_history' in data:
            return data['loss_history'].tolist()
        return None