from types import SimpleNamespace

args = SimpleNamespace(
    mode='train',
    seed=2024,
    data_dir='./data',
    preload=0,
    embed_dim=64,
    l2loss_lambda=1e-5,
    train_batch_size=1024,
    test_batch_size=1024,
    lr=0.0001,
    n_epoch=50,
    stopping_steps=2,
    checkpoint_every=5,
    evaluate_every=50,
    Ks='[1, 5, 10]',
    save_dir='./logs',
    device='cpu'
)