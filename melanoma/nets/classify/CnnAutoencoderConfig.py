from melanoma.constants.Constants import (EPOCH_NUMBER, BATCH_SIZE, HEIGHT, WIDTH, CHANNELS, MODEL_NAME, LEARNING_RATE, MODEL_DIR, CHECKPOINTS,
                                          MODEL_SUMMARY_DIR)

cnn_autoencoder_config = {
    EPOCH_NUMBER: 10,
    BATCH_SIZE: 2,

    HEIGHT: 200,
    WIDTH: 200,
    CHANNELS: 3,

    LEARNING_RATE: 0.001,
    MODEL_NAME: 'autoencoder',
    MODEL_DIR: '/home/pawols/auto',
    MODEL_SUMMARY_DIR: '/home/pawols/auto/summary',
    CHECKPOINTS: True
}
