SCORE_SCALE = 4
SCORE_SIZE = 5
VALID_LIMIT = 200
MLP_LATENT_DIM = 30
GMF_LATENT_DIM = 30
DROPOUT=0.0
FUSION_LAYERS = [ 60, 50, 40, 30]# make sure they are compatible with the
#latent dimension
LOGITS = 5
LOSS ='mse_loss' #'cross_entropy' 
GMF_FLAG = True
EPOCHS = 30
MAX_VALID = 10000
MODEL_NAME = "NMF"
