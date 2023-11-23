#include <iostream>
#include <vips/vips8>

using namespace std;
using namespace vips;

/* Convinience class to handle image manipulation */
class ImageContainer
{
public:
  VImage image;
  int width;
  int height;

  /* Given image path, initialize */
  void read_file(const string fname)
  {
    cout << "Read File got: " << fname << endl;

    ImageContainer::image = VImage::new_from_file(&fname[0], VImage::option()->set("access", VIPS_ACCESS_SEQUENTIAL));
    // cout << "image bands: " << ImageContainer::image.bands() << endl;
    // cout << "image format: " << ImageContainer::image.format() << endl;
    ImageContainer::width = ImageContainer::image.width();
    ImageContainer::height = ImageContainer::image.height();
  }

  /* get pixel value at (i, j) for band k */
  template <typename T>
  T get_pixel(int i, int j, int k){
    return (T)0;
  }
};
