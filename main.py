from load_data_from_cityscapes import load_data_from_cityscapes
from linknet import Linknet

def main():
  epochs = 100
  batch_size = 32
  label_path = "./labels"
  picture_path = "./pictures"
  height = 512
  width = 1024
  num_categories = 20
  linknet = Linknet(height,width,num_categories)
  print("finished loading models")
  x_train, y_train, x_test, y_test = load_data_from_cityscapes(picture_path, label_path)
  
  for epoch in range(epochs):
    print("epoch", epoch)
    linknet.train(x_train,y_train, batch_size)
    if(epoch % 10 == 0):
      linknet.test(x_test,y_test)
  
  linknet.sess.close()

if __name__ == '__main__':
  main()
