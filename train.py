import numpy as np

def compute_mse(w, r):
    return np.mean((w - r)**2)

def fit_gaussian(errors):
    mu_e = np.mean(errors)
    sigma_e = np.std(errors)
    return mu_e, sigma_e

def compute_threshold(mu_e, sigma_e, z_score=2.326):
    return mu_e + z_score * sigma_e

def evaluate(model, windows):
    errors = []
    for w in windows:
        r, _, _, _ = model.forward(w)
        errors.append(compute_mse(w, r))
    return np.array(errors)

def train(model, train_windows, epochs, lr):
    history = []
    M = len(train_windows)
    
    for epoch in range(epochs):
        total_loss = 0
        
        for i in range(M):
            w = train_windows[i]
            r, enc_caches, dec_caches, h_dec_seq = model.forward(w)
            
            loss = compute_mse(w, r)
            total_loss += loss
            
            dW_R = np.zeros_like(model.W_R)
            db_R = np.zeros_like(model.b_R)
            
            dh_dec = np.zeros((model.q, 1))
            dc_dec = np.zeros((model.q, 1))
            
            dW_dec = np.zeros_like(model.decoder.W)
            dU_dec = np.zeros_like(model.decoder.U)
            db_dec = np.zeros_like(model.decoder.b)
            
            dz_total = np.zeros((model.p, 1))
            
            for s in reversed(range(model.T)):
                err = (-2.0 / model.T) * (w[s] - r[s]).reshape(1, 1)
                
                dW_R += err @ h_dec_seq[s].T
                db_R += err
                
                dh_dec += model.W_R.T @ err
                
                dz_step, dh_dec, dc_dec, dW_d, dU_d, db_d = model.decoder.backward(dh_dec, dc_dec, dec_caches[s])
                
                dW_dec += dW_d
                dU_dec += dU_d
                db_dec += db_d
                dz_total += dz_step

            dh_enc = dz_total
            dc_enc = dc_dec
            
            dW_enc = np.zeros_like(model.encoder.W)
            dU_enc = np.zeros_like(model.encoder.U)
            db_enc = np.zeros_like(model.encoder.b)
            
            for t in reversed(range(model.T)):
                dx, dh_enc, dc_enc, dW_e, dU_e, db_e = model.encoder.backward(dh_enc, dc_enc, enc_caches[t])
                dW_enc += dW_e
                dU_enc += dU_e
                db_enc += db_e
                
            grads = (dW_enc, dU_enc, db_enc, dW_dec, dU_dec, db_dec, dW_R, db_R)
            model.update_params(grads, lr)
            
        history.append(total_loss / M)
        
    return history