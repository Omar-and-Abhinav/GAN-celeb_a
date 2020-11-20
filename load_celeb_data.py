import tensorflow_datasets as tfds
ds = tfds.load('celeb_a')
ds.download_and_prepare()


