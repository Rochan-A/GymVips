/* compile with:
 *      g++ -g -Wall example.cc `pkg-config vips-cpp --cflags --libs`
 */

#include <iostream>
#include <random>
#include <vector>
#include <utility>
#include <cstdlib>

#include "image_manip.hpp"

using namespace std;

/* Box-type for upper-left and lower-right coordinates */
typedef pair<pair<int, int>, pair<int, int>> box_t;

/* Function ported from Python to convert continuous coordinates of action to
  upper-left and lower-right coordinates given the image size and view size */
box_t continuous_to_coords(pair<float, float> action, pair<int, int> img_sz, pair<int, int> view_sz)
{
  float x = action.first;
  float y = action.second;

  x = (x + 1) / 2;
  y = (y + 1) / 2;

  pair<int, int> up_left{int((img_sz.first - view_sz.first) * x), int((img_sz.second - view_sz.second) * y)};
  pair<int, int> lower_right{up_left.first + view_sz.first, up_left.second + view_sz.second};

  return {up_left, lower_right};
}

/* BaseEnv class */
class BaseEnv
{

public:
  vector<string> files;
  pair<int, int> view_sz;
  ImageContainer ic;

  /* Initialize files that we need to read */
  BaseEnv(vector<string> file_paths, pair<int, int> view_sz)
  {
    BaseEnv::files = file_paths;
    BaseEnv::view_sz = view_sz;
  }

  /* Select a random file and initialize the ImageContainer */
  void _init_random_image()
  {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, files.size());

    ic.read_file(files[dis(gen)]);
    cout << "Read File!" << endl;
  }

  void get_region(int h, int w, int c, box_t patch)
  {
    return;
  }

  void reset()
  {
    _init_random_image();

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0f, 1.0f);

    /* random (x, y) coordinates */
    pair<float, float> points{float(dis(gen)), float(dis(gen))};
    box_t patch = continuous_to_coords(points, {ic.width, ic.height}, BaseEnv::view_sz);

    return;
  }

  void step(float action_x, float action_y)
  {
    /* Get box based on action */
    box_t patch = continuous_to_coords({action_x, action_y}, {ic.width, ic.height}, BaseEnv::view_sz);

    cout << "Step worked!" << endl;
    return;
  }
};


int
main (int argc, char **argv)
{ 
  if (VIPS_INIT (argv[0])) 
    vips_error_exit (NULL);

  vector<string> files{"/home/rochan/Documents/c_env/929cb4d0ec760882129e1853b335af17.tiff"};

  BaseEnv b(files, {256, 256});

  for (int i = 0; i < 5; i++){
    b.reset();
    b.step(0.2f, 0.2f);
  }

  vips_shutdown ();

  return 0;
}