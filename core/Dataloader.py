import random
import tensorflow as tf
import itertools
import cv2

class Dataloader(object):

    def __init__(self,df):
        self.df = df
    
    def generator(self):
        for i in itertools.count(1):
            img_path,bx,by,bh,bw = self.df.iloc[i]
            true_boxes = [bx,by,bh,bw]
            yield img_path,true_boxes

    def build_iterator(self):
        batch_size = 64
        prefetch_batch_buffer = 5

        dataset = tf.data.Dataset.from_generator(self.generator,(tf.string,tf.float32), 
                                                 (tf.TensorShape([]), tf.TensorShape([None])))
        dataset = dataset.map(self._read_image_and_resize)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(100)
        dataset = dataset.prefetch(prefetch_batch_buffer)
        itere  = dataset.make_one_shot_iterator()
        images,true_boxes = itere.get_next()
        
        return (images,true_boxes)
    
#     def generate_train_batch(self):
#         '''Yield a generator of training data from filename on given list of cols split for train/test'''
#         i = 0
#         while i < (12000):
#             x_batch = []
#             y_batch = []
#             for b in range(batch_size):
#                 if i >= (self.len_train - seq_len):
#                     # stop-condition for a smaller final batch if data doesn't divide evenly
#                     yield np.array(x_batch), np.array(self._encoding(y_batch))
#                     i = 0
#                 x, y = self._next_window(i, seq_len, normalise)
#                 x_batch.append(x)
#                 y_batch.extend(y)
#                 i += 1
#             yield np.array(x_batch), np.array(self._encoding(y_batch))
                
        
    def _read_image_and_resize(self,img_path,true_boxes):
        target_size = [480, 480]
        # read images from disk
        img1_file = tf.read_file(img_path)
        img1 = tf.image.decode_image(img1_file)
        img1 = tf.divide(img1,255)
        # let tensorflow know that the loaded images have unknown dimensions, and 3 color channels (rgb)
        img1.set_shape([None, None, 3])
        # resize to model input size
        img1_resized = tf.image.resize_images(img1, target_size)
        return img1_resized,true_boxes  