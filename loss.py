import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

class LossHelper:
    def __init__(self):

        # record losses through steps
        self.loss_d = tf.keras.metrics.Mean() # diffuse
        self.loss_s = tf.keras.metrics.Mean() # specular
        self.loss_g = tf.keras.metrics.Mean() # glossiness
        self.loss_n = tf.keras.metrics.Mean() # normal

        self.loss_i = tf.keras.metrics.Mean() # images

    def svbrdf(self, pred, target):

        # loss functions
        # MAE for diffuse, specular and glossiness
        # Cosine loss for normal
        loss_d = tf.keras.losses.MAE(target['diffuse'], pred['diffuse'])
        loss_s = tf.keras.losses.MAE(target['specular'], pred['specular'])
        loss_g = tf.keras.losses.MAE(target['glossiness'], pred['glossiness'])
        loss_n = tf.keras.losses.cosine_similarity(target['normal'], pred['normal']) + 1.0

        # keep the history
        self.loss_d(loss_d)
        self.loss_s(loss_s)
        self.loss_g(loss_g)
        self.loss_n(loss_n)

        # total loss = diffuse + specular + glossiness + normal
        return loss_d + loss_s + loss_g + loss_n

    # MAE loss for images
    def images(self, pred, target):
        loss_i = tf.keras.losses.MAE(target, pred)
        self.loss_i(loss_i)
        return loss_i

    # write loss summary to log
    def summary(self, prefix, step):        
        tf.summary.scalar(prefix + "-svbrdf/diffuse",   self.loss_d.result(), step=step)
        tf.summary.scalar(prefix + "-svbrdf/specular",  self.loss_s.result(), step=step)
        tf.summary.scalar(prefix + "-svbrdf/glossiness", self.loss_g.result(), step=step)
        tf.summary.scalar(prefix + "-svbrdf/normal",    self.loss_n.result(), step=step)
        tf.summary.scalar(prefix + "-images/images",  self.loss_i.result(), step=step)

    # get current mean loss
    def get_svbrdf_loss(self):
        return self.loss_d.result() + self.loss_s.result() + self.loss_g.result() + self.loss_n.result()
    
    def get_images_loss(self):
        return self.loss_i.result()

    # reset loss history
    def reset(self):        
        self.loss_d.reset_states()
        self.loss_s.reset_states()
        self.loss_g.reset_states()
        self.loss_n.reset_states()
        self.loss_i.reset_states()