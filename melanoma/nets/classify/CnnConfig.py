from melanoma.constants.Constants import (EPOCH_NUMBER, BATCH_SIZE, HEIGHT, WIDTH, CHANNELS, CLASS_NUMBER,
                                          DROPOUT_KEEP_PROB, MODEL_NAME, LEARNING_RATE, MODEL_DIR, CHECKPOINTS,
                                          MODEL_SUMMARY_DIR)

model_config = {
    EPOCH_NUMBER: 100,
    BATCH_SIZE: 256,

    HEIGHT: 28,
    WIDTH: 28,
    CHANNELS: 1,
    CLASS_NUMBER: 10,

    LEARNING_RATE: 0.001,
    DROPOUT_KEEP_PROB: 0.5,
    MODEL_NAME: 'mnist',
    MODEL_DIR: '/home/pawols/tmp_model_experiment',
    MODEL_SUMMARY_DIR: '/home/pawols/tmp_model_experiment/summary',
    CHECKPOINTS: True
}
