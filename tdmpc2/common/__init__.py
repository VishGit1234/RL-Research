MODEL_SIZE = { # parameters (M)
	1:   {'enc_dim': 256,
		  'mlp_dim': 384,
		  'latent_dim': 128,
		  'num_enc_layers': 2,
		  'num_q': 2},
	5:   {'enc_dim': 256,
		  'mlp_dim': 512,
		  'latent_dim': 512,
		  'num_enc_layers': 2},
	19:  {'enc_dim': 1024,
		  'mlp_dim': 1024,
		  'latent_dim': 768,
		  'num_enc_layers': 3},
	48:  {'enc_dim': 1792,
		  'mlp_dim': 1792,
		  'latent_dim': 768,
		  'num_enc_layers': 4},
	317: {'enc_dim': 4096,
		  'mlp_dim': 4096,
		  'latent_dim': 1376,
		  'num_enc_layers': 5,
		  'num_q': 8},
}