import pyglet
import ratcave as rc
import cv2 as cv
from skimage import io


class Monkey_Model:

  def __init__(self, window):
    self.rendered = False
    self.loaded = False

    self.scene = None
    self.window = window

    self.img = None
    self.img_name = 'monkey'


  def make_scene(self):

    # Insert filename into WavefrontReader.
    obj_filename = rc.resources.obj_primitives
    obj_reader = rc.WavefrontReader(obj_filename)

    # Create Mesh
    monkey = obj_reader.get_mesh("Monkey")
    monkey.position.xyz = 0, 0, -5

    # Create Scene
    self.scene = rc.Scene(meshes=[monkey], bgColor=(0, 1, 0))


  def draw(self):

    if not self.rendered:
      with rc.default_shader:

        #print(window.screen)
        self.scene.draw()

        # save image
        pyglet.image.get_buffer_manager().get_color_buffer().save(self.img_name)

        # set as rendered
        self.rendered = True



  def load_image(self):

    # read only if rendered first
    if self.rendered and not self.loaded:
      
      # read image
      img = cv.imread(self.img_name)

      # add alpha channel
      img = cv.cvtColor(img, cv.COLOR_RGB2RGBA)

      # alpha channel to zero
      img[img[:, :, 1] == 255] = 0

      self.img = img

      self.rendered = False
      self.loaded = True

      #io.imshow(img)
      #io.show()
      cv.imwrite(self.img_name + '_transp.png', img)

    else:
      print("must be rendered first")



if __name__ == '__main__':

  # Create Window
  window = pyglet.window.Window()

  monkey = Monkey_Model(window)
  monkey.make_scene()

  monkey.draw()
  monkey.load_image()


  #pyglet.app.run()


