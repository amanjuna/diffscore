'''
run_training.py

Creates a tensorflow session and actually
runs the training loop
'''
import tensorflow as tf
import numpy as np

## Temp imports for testing
import config
import Model


def train(train_model, dev_model, config):
    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_model.iter_init_op)
        
        train_writer = tf.summary.FileWriter('../tensorboard/train/')
        val_writer = tf.summary.FileWriter('../tensorboard/val/')

        best_dev = float('inf')
        for epoch in range(train_model.config.num_epochs):
            print("Epoch {}\n".format(epoch+1))
            num_steps = (config.train_size + config.batch_size - 1) // config.batch_size

            for t in range(num_steps):
                prediction, loss, curr_step = train_step(train_model, config, sess, train_writer)
                if t % 10 == 0:
                    print("Iteration {} loss: {}".format(t, loss))


            if epoch % config.save_every == 0:
                loss = val_epoch(dev_model, config, sess, val_writer)
                    
                if loss < best_dev:
                    best_dev = loss
                    saver.save(sess, config.checkpoints)
                print("Dev Loss: {}".format(best_dev))


def val_epoch(model, config, sess, val_writer):

    num_steps = (config.dev_size + config.batch_size - 1) // config.batch_size
    sess.run(model.iter_init_op)
    losses = []
    psnrs = []
    ssims = []
    for _ in range(num_steps):
        loss, psnr, ssim = sess.run([model.loss_op, model.psnr_op, model.ssim_op])
        losses.append(loss)
        psnrs.append(psnr)
        ssims.append(ssim)
    loss = np.mean(losses)
    psnr = np.mean(psnrs)
    ssim = np.mean(ssims)
    with tf.variable_scope("metrics"):
        summary = tf.Summary()
        summary.value.add(tag="metrics/Peak_Signal-to-Noise_Ratio", simple_value=psnr)
        summary.value.add(tag="metrics/Structural_Similarity", simple_value=ssim)
        summary.value.add(tag="metrics/Loss", simple_value=loss)
        
    val_writer.add_summary(summary, curr_step)
    return loss


def train_step(model, config, sess, train_writer):
    global_step = tf.train.get_global_step()
    _, high_res, loss, summary, global_step = sess.run([model.train_op, model.prediction_op, 
                                                        model.loss_op, model.merged, 
                                                        global_step])
    train_writer.add_summary(summary, global_step)
    return high_res, loss, global_step

if __name__=="__main__":
    all_data = pd.read_csv("../data/unified_processed.csv", index_col="Dataset")
    all_data = all_data.loc[:, "Standardized_Order":"weight"]
    for dset in config.ALLDATA[0]:
        
