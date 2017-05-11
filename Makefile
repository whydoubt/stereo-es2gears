# For fedora, dependencies include mesa-libGLES-devel and mesa-libgbm-devel

CFLAGS=-g -O2 -Wall -Wextra -fsanitize=address
DRM_FLAGS=`pkg-config --cflags --libs libdrm`

stereo-es2gears: stereo-es2gears.c
	$(CC) $(CFLAGS) $< -o $@ -lm $(DRM_FLAGS) -lgbm -lEGL -lGLESv2

clean:
	rm -f stereo-es2gears
