/*
 * Copyright (C) 1999-2001  Brian Paul   All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * BRIAN PAUL BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
 * AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/*
 * Ported to GLES2.
 * Kristian Høgsberg <krh@bitplanet.net>
 * May 3, 2010
 *
 * Improve GLES2 port:
 *   * Refactor gear drawing.
 *   * Use correct normals for surfaces.
 *   * Improve shader.
 *   * Use perspective projection transformation.
 *   * Add FPS count.
 *   * Add comments.
 * Alexandros Frantzis <alexandros.frantzis@linaro.org>
 * Jul 13, 2010
 */

/*
 * Converted to stereoscopic 3d.
 * Neil Roberts <neil@linux.intel.com>
 * 2013, 2014
 * Based on the modesetting example written in 2012
 *  by David Herrmann <dh.herrmann@googlemail.com>
 */

#define _GNU_SOURCE

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <errno.h>
#include <fcntl.h>
#include <gbm.h>
#include <math.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <assert.h>

struct mode_layout {
   /* Total size of the buffer containing the combined images */
   uint32_t buffer_width, buffer_height;
   /* Actual size in pixels of each eye */
   uint32_t eye_width, eye_height;
   /* Virtual size that each eye will be displayed at. This is needed because
    * certain modes use non-square pixels and squash two images into one. The
    * TV then effectively scales them back up to this size */
   uint32_t virtual_eye_width, virtual_eye_height;
   /* Offset in pixels to the position of the right eye within the buffer */
   uint32_t right_eye_x, left_eye_y;
};

struct gbm_dev {
   int fd;
   struct mode_layout layout;

   drmModeModeInfo mode;
   uint32_t conn;
   uint32_t crtc;
   drmModeCrtc *saved_crtc;

   int pending_swap;
};

struct gbm_context {
   struct gbm_dev *dev;
   struct gbm_device *gbm;
   struct gbm_surface *gbm_surface;
   EGLDisplay edpy;
   EGLConfig egl_config;
   EGLSurface egl_surface;
   EGLContext egl_context;

   uint32_t current_fb_id;
   struct gbm_bo *current_bo;
};

struct stereo_options {
   const char *card;
   const char *stereo_layout;
   uint32_t connector;
};

struct stereo_winsys {
   int fd;
   struct gbm_dev *dev;
   struct gbm_context *context;
};

struct stereo_renderer {
   struct mode_layout layout;
};

struct stereo_data {
   struct stereo_winsys *winsys;
   struct stereo_renderer *renderer;
};

struct stereo_mode {
   int mode_number;
   const char *short_name;
   const char *long_name;
};

static const struct stereo_mode
stereo_modes[] = {
   { DRM_MODE_FLAG_3D_NONE, "none", "none" },
   { DRM_MODE_FLAG_3D_FRAME_PACKING, "fp", "frame packing" },
   { DRM_MODE_FLAG_3D_FIELD_ALTERNATIVE, "fa", "field alternative" },
   { DRM_MODE_FLAG_3D_LINE_ALTERNATIVE, "la", "line alternative" },
   { DRM_MODE_FLAG_3D_SIDE_BY_SIDE_FULL, "sbsf", "side by side full" },
   { DRM_MODE_FLAG_3D_L_DEPTH, "ld", "l depth" },
   { DRM_MODE_FLAG_3D_L_DEPTH_GFX_GFX_DEPTH, "ldggd", "l depth gfx gfx depth" },
   { DRM_MODE_FLAG_3D_TOP_AND_BOTTOM, "tb", "top and bottom" },
   { DRM_MODE_FLAG_3D_SIDE_BY_SIDE_HALF, "sbsh", "side by side half" },
};

#define STRIPS_PER_TOOTH 7
#define VERTICES_PER_TOOTH 34
#define GEAR_VERTEX_STRIDE 6

#define UNUSED(x) (void)(x)

/**
 * Struct describing the vertices in triangle strip
 */
struct vertex_strip {
   /** The first vertex in the strip */
   GLint first;
   /** The number of consecutive vertices in the strip after the first */
   GLint count;
};

/* Each vertex consist of GEAR_VERTEX_STRIDE GLfloat attributes */
typedef GLfloat GearVertex[GEAR_VERTEX_STRIDE];

/**
 * Struct representing a gear.
 */
struct gear {
   /** The array of vertices comprising the gear */
   GearVertex *vertices;
   /** The number of vertices comprising the gear */
   int nvertices;
   /** The array of triangle strips comprising the gear */
   struct vertex_strip *strips;
   /** The number of triangle strips comprising the gear */
   int nstrips;
   /** The Vertex Buffer Object holding the vertices in the
    * graphics card */
   GLuint vbo;
};

/** The view rotation [x, y, z] */
static GLfloat view_rot[3] = { 50.0, 30.0, 0.0 };

/** The gears */
static struct gear *gear1, *gear2, *gear3;
/** The current gear rotation angle */
static GLfloat angle = 0.0;
/** The location of the shader uniforms */
static GLuint ModelViewProjectionMatrix_location,
   NormalMatrix_location, LightSourcePosition_location,
   MaterialColor_location;
/** The projection matrix */
static GLfloat ProjectionMatrix[16];
/** The direction of the directional light for the scene */
static const GLfloat LightSourcePosition[4] = { 5.0, 5.0, 10.0, 1.0 };

static GLfloat eyesep = 0.5;            /* Eye separation. */
static GLfloat fix_point = 40.0;        /* Fixation point distance.  */
static GLfloat left, right, asp;        /* Stereo frustum params.  */

static int quit = 0;

static void *
xmalloc(size_t size)
{
   void *res = malloc(size);

   if (res)
      return res;

   abort();
}

static int
stereo_find_crtc(drmModeRes *res, drmModeConnector *conn,
                 struct gbm_dev *dev)
{
   drmModeEncoder *enc;
   int i, j;
   uint32_t crtc;

   /* first try the currently conected encoder+crtc */
   if (conn->encoder_id) {
      enc = drmModeGetEncoder(dev->fd, conn->encoder_id);
      if (enc->crtc_id > 0) {
         drmModeFreeEncoder(enc);
         dev->crtc = enc->crtc_id;
         return 0;
      }
   }

   /* If the connector is not currently bound to an encoder
    * iterate all other available encoders to find a matching
    * CRTC. */
   for (i = 0; i < conn->count_encoders; ++i) {
      enc = drmModeGetEncoder(dev->fd, conn->encoders[i]);
      if (!enc) {
         fprintf(stderr,
                 "cannot retrieve encoder %u:%u (%d): %m\n",
                 i, conn->encoders[i], errno);
         continue;
      }

      /* iterate all global CRTCs */
      for (j = 0; j < res->count_crtcs; ++j) {
         /* check whether this CRTC works with the encoder */
         if (!(enc->possible_crtcs & (1 << j)))
            continue;

         /* check that no other device already uses this CRTC */
         crtc = res->crtcs[j];

         /* we have found a CRTC, so save it and return */
         if (crtc > 0) {
            drmModeFreeEncoder(enc);
            dev->crtc = crtc;
            return 0;
         }
      }

      drmModeFreeEncoder(enc);
   }

   fprintf(stderr, "cannot find suitable CRTC for connector %u\n",
           conn->connector_id);
   return -ENOENT;
}

static const struct stereo_mode *
get_stereo_mode(int mode_number)
{
   unsigned int i;

   for (i = 0; i < sizeof stereo_modes / sizeof stereo_modes[0]; i++)
      if (stereo_modes[i].mode_number == mode_number)
         return stereo_modes + i;

   return NULL;
}

static int
get_mode_rank(const drmModeModeInfo *mode)
{
   static const int ranks[] = {
      DRM_MODE_FLAG_3D_NONE,
      /* These two modes have half a frame for each eye and end up with
       * non-square pixels */
      DRM_MODE_FLAG_3D_SIDE_BY_SIDE_HALF,
      DRM_MODE_FLAG_3D_TOP_AND_BOTTOM,
      /* These two modes have a complete frame for each eye */
      DRM_MODE_FLAG_3D_SIDE_BY_SIDE_FULL,
      DRM_MODE_FLAG_3D_FRAME_PACKING,
   };
   int layout;
   unsigned int i;

   if (mode == NULL)
      return -1;

   layout = mode->flags & DRM_MODE_FLAG_3D_MASK;

   for (i = 0; i < sizeof(ranks) / sizeof(ranks[0]); i++)
      if (ranks[i] == layout)
         return i;

   return -1;
}

static bool
is_chosen_mode(const drmModeModeInfo *mode,
               const struct stereo_options *options,
               const drmModeModeInfo *old_mode)
{
   const struct stereo_mode *stereo_mode;
   int mode_rank;

   stereo_mode = get_stereo_mode(mode->flags & DRM_MODE_FLAG_3D_MASK);

   if (options->stereo_layout &&
       strcmp(stereo_mode->short_name, options->stereo_layout))
      return false;

   mode_rank = get_mode_rank(mode);

   return mode_rank != -1 && mode_rank > get_mode_rank(old_mode);
}

static int
find_mode(struct gbm_dev *dev, drmModeConnector *conn,
          const struct stereo_options *options)
{
   const drmModeModeInfo *old_mode = NULL;
   int i;

   for (i = 0; i < conn->count_modes; i++) {
      if (is_chosen_mode(conn->modes + i, options, old_mode)) {
         dev->mode = conn->modes[i];
         old_mode = &conn->modes[i];
      }
   }

   return old_mode ? 0 : -ENOENT;
}

static void
get_layout_for_mode(struct mode_layout *layout,
                    const drmModeModeInfo *mode)
{
   switch (mode->flags & DRM_MODE_FLAG_3D_MASK) {
   case DRM_MODE_FLAG_3D_NONE:
      layout->buffer_width = mode->hdisplay;
      layout->buffer_height = mode->vdisplay;
      layout->eye_width = layout->buffer_width;
      layout->eye_height = layout->buffer_height;
      layout->virtual_eye_width = layout->eye_width;
      layout->virtual_eye_height = layout->eye_height;
      /* make the right eye off the screen to get rid of it */
      layout->right_eye_x = layout->eye_width;
      layout->left_eye_y = 0;
      break;
   case DRM_MODE_FLAG_3D_SIDE_BY_SIDE_HALF:
      layout->buffer_width = mode->hdisplay;
      layout->buffer_height = mode->vdisplay;
      layout->eye_width = mode->hdisplay / 2;
      layout->eye_height = mode->vdisplay;
      layout->virtual_eye_width = layout->buffer_width;
      layout->virtual_eye_height = layout->eye_height;
      layout->right_eye_x = layout->eye_width;
      layout->left_eye_y = 0;
      break;
   case DRM_MODE_FLAG_3D_SIDE_BY_SIDE_FULL:
      layout->buffer_width = mode->hdisplay * 2;
      layout->buffer_height = mode->vdisplay;
      layout->eye_width = mode->hdisplay;
      layout->eye_height = mode->vdisplay;
      layout->virtual_eye_width = layout->eye_width;
      layout->virtual_eye_height = layout->eye_height;
      layout->right_eye_x = layout->eye_width;
      layout->left_eye_y = 0;
      break;
   case DRM_MODE_FLAG_3D_TOP_AND_BOTTOM:
      layout->buffer_width = mode->hdisplay;
      layout->buffer_height = mode->vdisplay;
      layout->eye_width = mode->hdisplay;
      layout->eye_height = mode->vdisplay / 2;
      layout->virtual_eye_width = layout->eye_width;
      layout->virtual_eye_height = layout->buffer_height;
      layout->right_eye_x = 0;
      layout->left_eye_y = layout->eye_height;
      break;
   case DRM_MODE_FLAG_3D_FRAME_PACKING:
      layout->buffer_width = mode->hdisplay;
      layout->buffer_height = mode->vtotal + mode->vdisplay;
      layout->eye_width = mode->hdisplay;
      layout->eye_height = mode->vdisplay;
      layout->virtual_eye_width = layout->eye_width;
      layout->virtual_eye_height = layout->eye_height;
      layout->right_eye_x = 0;
      layout->left_eye_y = mode->vtotal;
      break;
   default:
      assert(0);
   }
}

static int
stereo_setup_dev(drmModeRes *res, drmModeConnector *conn,
                 const struct stereo_options *options,
                 struct gbm_dev *dev)
{
   int mode_3d;
   int ret;

   /* check if a monitor is connected */
   if (conn->connection != DRM_MODE_CONNECTED) {
      fprintf(stderr, "ignoring unused connector %u\n",
              conn->connector_id);
      return -ENOENT;
   }

   ret = find_mode(dev, conn, options);
   if (ret) {
      fprintf(stderr, "no valid mode for connector %u\n",
              conn->connector_id);
      return ret;
   }

   get_layout_for_mode(&dev->layout, &dev->mode);

   mode_3d = dev->mode.flags & DRM_MODE_FLAG_3D_MASK;

   fprintf(stderr, "mode for connector %u is %ux%u (%s)\n",
           conn->connector_id,
           dev->layout.eye_width, dev->layout.eye_height,
           get_stereo_mode(mode_3d)->long_name);

   if (mode_3d == DRM_MODE_FLAG_3D_NONE)
      fprintf(stderr, "WARNING: no usable stereoscopic mode was found, "
              "rendering in 2D\n");

   /* find a crtc for this connector */
   ret = stereo_find_crtc(res, conn, dev);
   if (ret) {
      fprintf(stderr, "no valid crtc for connector %u\n",
              conn->connector_id);
      return ret;
   }

   return 0;
}

static int
stereo_open(int *out, const struct stereo_options *options)
{
   const char *card = options->card;
   int fd, ret;

   if (card == NULL)
      card = "/dev/dri/card0";

   fd = open(card, O_RDWR | O_CLOEXEC);
   if (fd < 0) {
      ret = -errno;
      fprintf(stderr, "cannot open '%s': %m\n", card);
      return ret;
   }

   if (drmSetClientCap(fd, DRM_CLIENT_CAP_STEREO_3D, 1)) {
      fprintf(stderr, "error setting stereo client cap: %m\n");
      close(fd);
      return -errno;
   }

   *out = fd;
   return 0;
}

static drmModeConnector *
get_connector(int fd, drmModeRes *res,
              const struct stereo_options *options)
{
   drmModeConnector *conn;
   int i;

   for (i = 0; i < res->count_connectors; i++) {
      conn = drmModeGetConnector(fd, res->connectors[i]);

      if (conn == NULL) {
         fprintf(stderr,
                 "cannot retrieve DRM connector "
                 "%u:%u (%d): %m\n",
                 i, res->connectors[i], errno);
         return NULL;
      }

      if (options->connector == 0 ||
          conn->connector_id == options->connector)
         return conn;
      drmModeFreeConnector(conn);
   }

   fprintf(stderr,
           "couldn't find connector with id %u\n",
           options->connector);

   return NULL;
}

static struct gbm_dev *
stereo_prepare_dev(int fd, const struct stereo_options *options)
{
   drmModeRes *res;
   drmModeConnector *conn;
   struct gbm_dev *dev;
   int ret;

   /* retrieve resources */
   res = drmModeGetResources(fd);
   if (!res) {
      fprintf(stderr, "cannot retrieve DRM resources (%d): %m\n",
              errno);
      goto error;
   }

   conn = get_connector(fd, res, options);
   if (!conn)
      goto error_resources;

   /* create a device structure */
   dev = xmalloc(sizeof(*dev));
   memset(dev, 0, sizeof(*dev));
   dev->conn = conn->connector_id;
   dev->fd = fd;

   /* call helper function to prepare this connector */
   ret = stereo_setup_dev(res, conn, options, dev);
   if (ret) {
      if (ret != -ENOENT) {
         errno = -ret;
         fprintf(stderr,
                 "cannot setup device for connector "
                 "%u:%u (%d): %m\n",
                 0, res->connectors[0], errno);
      }
      goto error_dev;
   }

   drmModeFreeConnector(conn);
   drmModeFreeResources(res);

   return dev;

error_dev:
   free(dev);
   drmModeFreeConnector(conn);
error_resources:
   drmModeFreeResources(res);
error:
   return NULL;
}

static void
restore_saved_crtc(struct gbm_dev *dev)
{
   /* restore saved CRTC configuration */
   if (dev->saved_crtc) {
      drmModeSetCrtc(dev->fd,
                     dev->saved_crtc->crtc_id,
                     dev->saved_crtc->buffer_id,
                     dev->saved_crtc->x,
                     dev->saved_crtc->y,
                     &dev->conn,
                     1,
                     &dev->saved_crtc->mode);
      drmModeFreeCrtc(dev->saved_crtc);

      dev->saved_crtc = NULL;
   }
}

static void
stereo_cleanup_dev(struct gbm_dev *dev)
{
   restore_saved_crtc(dev);

   /* free allocated memory */
   free(dev);
}

static void
free_current_bo(struct gbm_context *context)
{
   if (context->current_fb_id) {
      drmModeRmFB(context->dev->fd, context->current_fb_id);
      context->current_fb_id = 0;
   }
   if (context->current_bo) {
      gbm_surface_release_buffer(context->gbm_surface,
                                 context->current_bo);
      context->current_bo = NULL;
   }
}

static int
create_gbm_surface(struct gbm_context *context)
{
   const uint32_t flags = GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING;

   context->gbm_surface = gbm_surface_create(context->gbm,
                                             context->dev->layout.buffer_width,
                                             context->dev->layout.buffer_height,
                                             GBM_BO_FORMAT_XRGB8888,
                                             flags);

   if (context->gbm_surface == NULL) {
      fprintf(stderr, "error creating GBM surface\n");
      return -ENOENT;
   }

   return 0;
}

static int
choose_egl_config(struct gbm_context *context)
{
   static const EGLint attribs[] = {
      EGL_RED_SIZE, 1,
      EGL_GREEN_SIZE, 1,
      EGL_BLUE_SIZE, 1,
      EGL_ALPHA_SIZE, EGL_DONT_CARE,
      EGL_DEPTH_SIZE, 1,
      EGL_BUFFER_SIZE, EGL_DONT_CARE,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
      EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
      EGL_NONE
   };
   EGLBoolean status;
   EGLint config_count;

   status = eglChooseConfig(context->edpy,
                            attribs,
                            &context->egl_config, 1,
                            &config_count);
   if (status != EGL_TRUE || config_count < 1) {
      fprintf(stderr, "Unable to find a usable EGL configuration\n");
      return -ENOENT;
   }

   return 0;
}

static int
create_egl_surface(struct gbm_context *context)
{
   context->egl_surface =
      eglCreateWindowSurface(context->edpy,
                             context->egl_config,
                             (NativeWindowType) context->gbm_surface,
                             NULL);
   if (context->egl_surface == EGL_NO_SURFACE) {
      fprintf(stderr, "Failed to create EGL surface\n");
      return -ENOENT;
   }

   return 0;
}

static int
create_egl_context(struct gbm_context *context)
{
   static const EGLint attribs[] = {
      EGL_CONTEXT_CLIENT_VERSION, 2,
      EGL_NONE
   };

   context->egl_context = eglCreateContext(context->edpy,
                                           context->egl_config,
                                           EGL_NO_CONTEXT,
                                           attribs);
   if (context->egl_context == EGL_NO_CONTEXT) {
      fprintf(stderr, "Error creating EGL context\n");
      return -ENOENT;
   }

   return 0;
}

static struct gbm_context *
stereo_prepare_context(struct gbm_dev *dev)
{
   struct gbm_context *context;

   context = xmalloc(sizeof(*context));
   context->dev = dev;

   context->gbm = gbm_create_device(dev->fd);
   if (context->gbm == NULL) {
      fprintf(stderr, "error creating GBM device\n");
      goto error;
   }

   context->edpy = eglGetDisplay((EGLNativeDisplayType) context->gbm);
   if (context->edpy == EGL_NO_DISPLAY) {
      fprintf(stderr, "error getting EGL display\n");
      goto error_gbm_device;
   }

   if (!eglInitialize(context->edpy, NULL, NULL)) {
      fprintf(stderr, "error intializing EGL display\n");
      goto error_gbm_device;
   }

   if (create_gbm_surface(context))
      goto error_egl_display;

   if (choose_egl_config(context))
      goto error_gbm_surface;

   if (create_egl_surface(context))
      goto error_gbm_surface;

   if (create_egl_context(context))
      goto error_egl_surface;

   if (!eglMakeCurrent(context->edpy,
                       context->egl_surface,
                       context->egl_surface,
                       context->egl_context)) {
      fprintf(stderr, "failed to make EGL context current\n");
      goto error_egl_context;
   }

   return context;

error_egl_context:
   eglDestroyContext(context->edpy, context->egl_context);
error_egl_surface:
   eglDestroySurface(context->edpy, context->egl_surface);
error_gbm_surface:
   gbm_surface_destroy(context->gbm_surface);
error_egl_display:
   eglTerminate(context->edpy);
error_gbm_device:
   gbm_device_destroy(context->gbm);
error:
   free(context);
   return NULL;
}

static void
stereo_cleanup_context(struct gbm_context *context)
{
   restore_saved_crtc(context->dev);
   free_current_bo(context);
   eglMakeCurrent(context->edpy,
                  EGL_NO_SURFACE,
                  EGL_NO_SURFACE,
                  EGL_NO_CONTEXT);
   eglDestroyContext(context->edpy, context->egl_context);
   eglDestroySurface(context->edpy, context->egl_surface);
   gbm_surface_destroy(context->gbm_surface);
   eglTerminate(context->edpy);
   gbm_device_destroy(context->gbm);
   free(context);
}

static void
page_flip_handler(int fd,
                  unsigned int frame,
                  unsigned int sec,
                  unsigned int usec,
                  void *data)
{
   UNUSED(fd);
   UNUSED(frame);
   UNUSED(sec);
   UNUSED(usec);

   struct gbm_dev *dev = data;

   dev->pending_swap = 0;
}

static void
wait_swap(struct gbm_dev *dev)
{
   drmEventContext evctx;

   while (dev->pending_swap) {
      memset(&evctx, 0, sizeof(evctx));
      evctx.version = DRM_EVENT_CONTEXT_VERSION;
      evctx.page_flip_handler = page_flip_handler;
      drmHandleEvent(dev->fd, &evctx);
   }
}

static int
set_initial_crtc(struct gbm_dev *dev, uint32_t fb_id)
{
   dev->saved_crtc = drmModeGetCrtc(dev->fd, dev->crtc);

   if (drmModeSetCrtc(dev->fd,
                      dev->crtc,
                      fb_id,
                      0, 0, /* x/y */
                      &dev->conn, 1,
                      &dev->mode)) {
      fprintf(stderr, "Failed to set drm mode: %m\n");
      return errno;
   }

   return 0;
}

static void
swap(struct stereo_winsys *winsys)
{
   struct gbm_dev *dev = winsys->dev;
   struct gbm_context *context = winsys->context;
   struct gbm_bo *bo;
   uint32_t handle, stride;
   uint32_t width, height;
   uint32_t fb_id;

   eglSwapBuffers(context->edpy, context->egl_surface);

   bo = gbm_surface_lock_front_buffer(context->gbm_surface);
   width = gbm_bo_get_width(bo);
   height = gbm_bo_get_height(bo);
   stride = gbm_bo_get_stride(bo);
   handle = gbm_bo_get_handle(bo).u32;

   if (drmModeAddFB(dev->fd,
                    width, height,
                    24, /* depth */
                    32, /* bpp */
                    stride,
                    handle,
                    &fb_id)) {
      fprintf(stderr,
              "Failed to create new back buffer handle: %m\n");
   } else {
      if (dev->saved_crtc == NULL &&
          set_initial_crtc(dev, fb_id))
         return;

      if (drmModePageFlip(dev->fd,
                          dev->crtc,
                          fb_id,
                          DRM_MODE_PAGE_FLIP_EVENT,
                          dev)) {
         fprintf(stderr, "Failed to page flip: %m\n");
         return;
      }

      dev->pending_swap = 1;

      wait_swap(dev);

      free_current_bo(context);
      context->current_bo = bo;
      context->current_fb_id = fb_id;
   }
}

static void
winsys_disconnect(struct stereo_winsys *winsys)
{
   if (winsys->context) {
      stereo_cleanup_context(winsys->context);
      winsys->context = NULL;
   }
   if (winsys->dev) {
      stereo_cleanup_dev(winsys->dev);
      winsys->dev = NULL;
   }
   if (winsys->fd != -1) {
      close(winsys->fd);
      winsys->fd = -1;
   }
}

static int
winsys_connect(struct stereo_winsys *winsys,
               const struct stereo_options *options)
{
   int ret;

   /* open the DRM device */
   ret = stereo_open(&winsys->fd, options);
   if (ret)
      goto error;

   /* prepare all connectors and CRTCs */
   winsys->dev = stereo_prepare_dev(winsys->fd, options);
   if (winsys->dev == NULL) {
      ret = -ENOENT;
      goto error;
   }

   winsys->context = stereo_prepare_context(winsys->dev);
   if (winsys->context == NULL) {
      ret = -ENOENT;
      goto error;
   }

   return 0;

error:
   winsys_disconnect(winsys);
   return ret;
}

static void
winsys_free(struct stereo_winsys *winsys)
{
   winsys_disconnect(winsys);
   free(winsys);
}

static struct stereo_winsys *
create_winsys(const struct stereo_options *options)
{
   struct stereo_winsys *winsys = xmalloc(sizeof *winsys);

   memset(winsys, 0, sizeof *winsys);

   winsys->fd = -1;

   if (winsys_connect(winsys, options) != 0) {
      winsys_free(winsys);
      return NULL;
   }

   return winsys;
}

/**
 * Fills a gear vertex.
 *
 * @param v the vertex to fill
 * @param x the x coordinate
 * @param y the y coordinate
 * @param z the z coortinate
 * @param n pointer to the normal table
 *
 * @return the operation error code
 */
static GearVertex *
vert(GearVertex * v,
     GLfloat x, GLfloat y, GLfloat z,
     GLfloat n[3])
{
   v[0][0] = x;
   v[0][1] = y;
   v[0][2] = z;
   v[0][3] = n[0];
   v[0][4] = n[1];
   v[0][5] = n[2];

   return v + 1;
}

/**
 *  Create a gear wheel.
 *
 *  @param inner_radius radius of hole at center
 *  @param outer_radius radius at center of teeth
 *  @param width width of gear
 *  @param teeth number of teeth
 *  @param tooth_depth depth of tooth
 *
 *  @return pointer to the constructed struct gear
 */
static struct gear *
create_gear(GLfloat inner_radius, GLfloat outer_radius,
            GLfloat width, GLint teeth, GLfloat tooth_depth)
{
   GLfloat r0, r1, r2;
   GLfloat da;
   GearVertex *v;
   struct gear *gear;
   double s[5], c[5];
   GLfloat normal[3];
   int cur_strip = 0;
   int i;

   /* Allocate memory for the gear */
   gear = malloc(sizeof *gear);
   if (gear == NULL)
      return NULL;

   /* Calculate the radii used in the gear */
   r0 = inner_radius;
   r1 = outer_radius - tooth_depth / 2.0;
   r2 = outer_radius + tooth_depth / 2.0;

   da = 2.0 * M_PI / teeth / 4.0;

   /* Allocate memory for the triangle strip information */
   gear->nstrips = STRIPS_PER_TOOTH * teeth;
   gear->strips = calloc(gear->nstrips, sizeof(*gear->strips));

   /* Allocate memory for the vertices */
   gear->vertices =
      calloc(VERTICES_PER_TOOTH * teeth, sizeof(*gear->vertices));
   v = gear->vertices;

   for (i = 0; i < teeth; i++) {
      /* Calculate needed sin/cos for varius angles */
      sincos(i * 2.0 * M_PI / teeth, &s[0], &c[0]);
      sincos(i * 2.0 * M_PI / teeth + da, &s[1], &c[1]);
      sincos(i * 2.0 * M_PI / teeth + da * 2, &s[2], &c[2]);
      sincos(i * 2.0 * M_PI / teeth + da * 3, &s[3], &c[3]);
      sincos(i * 2.0 * M_PI / teeth + da * 4, &s[4], &c[4]);

      /* A set of macros for making the creation of the
       * gears easier */
#define  GEAR_POINT(r, da) { (r) * c[(da)], (r) * s[(da)] }
#define  SET_NORMAL(x, y, z) do {                               \
         normal[0] = (x); normal[1] = (y); normal[2] = (z);     \
      } while(0)

#define  GEAR_VERT(v, point, sign)                                      \
      vert((v), p[(point)].x, p[(point)].y, (sign) * width * 0.5, normal)

#define START_STRIP do {                                        \
         gear->strips[cur_strip].first = v - gear->vertices;    \
      } while(0);

#define END_STRIP do {                          \
         int _tmp = (v - gear->vertices);       \
         gear->strips[cur_strip].count = _tmp - \
            gear->strips[cur_strip].first;      \
         cur_strip++;                           \
      } while (0)

#define QUAD_WITH_NORMAL(p1, p2) do {                   \
         SET_NORMAL((p[(p1)].y - p[(p2)].y),            \
                    -(p[(p1)].x - p[(p2)].x), 0);       \
         v = GEAR_VERT(v, (p1), -1);                    \
         v = GEAR_VERT(v, (p1), 1);                     \
         v = GEAR_VERT(v, (p2), -1);                    \
         v = GEAR_VERT(v, (p2), 1);                     \
      } while(0)

      struct point {
         GLfloat x;
         GLfloat y;
      };

      /* Create the 7 points (only x,y coords) used to draw a tooth */
      struct point p[7] = {
         GEAR_POINT(r2, 1),      // 0
         GEAR_POINT(r2, 2),      // 1
         GEAR_POINT(r1, 0),      // 2
         GEAR_POINT(r1, 3),      // 3
         GEAR_POINT(r0, 0),      // 4
         GEAR_POINT(r1, 4),      // 5
         GEAR_POINT(r0, 4),      // 6
      };

      /* Front face */
      START_STRIP;
      SET_NORMAL(0, 0, 1.0);
      v = GEAR_VERT(v, 0, +1);
      v = GEAR_VERT(v, 1, +1);
      v = GEAR_VERT(v, 2, +1);
      v = GEAR_VERT(v, 3, +1);
      v = GEAR_VERT(v, 4, +1);
      v = GEAR_VERT(v, 5, +1);
      v = GEAR_VERT(v, 6, +1);
      END_STRIP;

      /* Inner face */
      START_STRIP;
      QUAD_WITH_NORMAL(4, 6);
      END_STRIP;

      /* Back face */
      START_STRIP;
      SET_NORMAL(0, 0, -1.0);
      v = GEAR_VERT(v, 6, -1);
      v = GEAR_VERT(v, 5, -1);
      v = GEAR_VERT(v, 4, -1);
      v = GEAR_VERT(v, 3, -1);
      v = GEAR_VERT(v, 2, -1);
      v = GEAR_VERT(v, 1, -1);
      v = GEAR_VERT(v, 0, -1);
      END_STRIP;

      /* Outer face */
      START_STRIP;
      QUAD_WITH_NORMAL(0, 2);
      END_STRIP;

      START_STRIP;
      QUAD_WITH_NORMAL(1, 0);
      END_STRIP;

      START_STRIP;
      QUAD_WITH_NORMAL(3, 1);
      END_STRIP;

      START_STRIP;
      QUAD_WITH_NORMAL(5, 3);
      END_STRIP;
   }

   gear->nvertices = (v - gear->vertices);

   /* Store the vertices in a vertex buffer object (VBO) */
   glGenBuffers(1, &gear->vbo);
   glBindBuffer(GL_ARRAY_BUFFER, gear->vbo);
   glBufferData(GL_ARRAY_BUFFER, gear->nvertices * sizeof(GearVertex),
                gear->vertices, GL_STATIC_DRAW);

   return gear;
}

/**
 * Multiplies two 4x4 matrices.
 *
 * The result is stored in matrix m.
 *
 * @param m the first matrix to multiply
 * @param n the second matrix to multiply
 */
static void
multiply(GLfloat * m, const GLfloat * n)
{
   GLfloat tmp[16];
   const GLfloat *row, *column;
   div_t d;
   int i, j;

   for (i = 0; i < 16; i++) {
      tmp[i] = 0;
      d = div(i, 4);
      row = n + d.quot * 4;
      column = m + d.rem;
      for (j = 0; j < 4; j++)
         tmp[i] += row[j] * column[j * 4];
   }
   memcpy(m, &tmp, sizeof tmp);
}

/**
 * Rotates a 4x4 matrix.
 *
 * @param[in,out] m the matrix to rotate
 * @param angle the angle to rotate
 * @param x the x component of the direction to rotate to
 * @param y the y component of the direction to rotate to
 * @param z the z component of the direction to rotate to
 */
static void
rotate(GLfloat * m, GLfloat angle, GLfloat x, GLfloat y, GLfloat z)
{
   double s, c;

   sincos(angle, &s, &c);
   GLfloat r[16] = {
      x * x * (1 - c) + c, y * x * (1 - c) + z * s,
      x * z * (1 - c) - y * s, 0,
      x * y * (1 - c) - z * s, y * y * (1 - c) + c,
      y * z * (1 - c) + x * s, 0,
      x * z * (1 - c) + y * s, y * z * (1 - c) - x * s,
      z * z * (1 - c) + c, 0,
      0, 0, 0, 1
   };

   multiply(m, r);
}

/**
 * Translates a 4x4 matrix.
 *
 * @param[in,out] m the matrix to translate
 * @param x the x component of the direction to translate to
 * @param y the y component of the direction to translate to
 * @param z the z component of the direction to translate to
 */
static void
translate(GLfloat * m, GLfloat x, GLfloat y, GLfloat z)
{
   GLfloat t[16] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1 };

   multiply(m, t);
}

/**
 * Creates an identity 4x4 matrix.
 *
 * @param m the matrix make an identity matrix
 */
static void
identity(GLfloat * m)
{
   GLfloat t[16] = {
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0,
   };

   memcpy(m, t, sizeof(t));
}

/**
 * Transposes a 4x4 matrix.
 *
 * @param m the matrix to transpose
 */
static void
transpose(GLfloat * m)
{
   GLfloat t[16] = {
      m[0], m[4], m[8], m[12],
      m[1], m[5], m[9], m[13],
      m[2], m[6], m[10], m[14],
      m[3], m[7], m[11], m[15]
   };

   memcpy(m, t, sizeof(t));
}

/**
 * Inverts a 4x4 matrix.
 *
 * This function can currently handle only pure translation-rotation matrices.
 * Read http://www.gamedev.net/community/forums/topic.asp?topic_id=425118
 * for an explanation.
 */
static void
invert(GLfloat * m)
{
   GLfloat t[16];
   identity(t);

   // Extract and invert the translation part 't'. The inverse of a
   // translation matrix can be calculated by negating the translation
   // coordinates.
   t[12] = -m[12];
   t[13] = -m[13];
   t[14] = -m[14];

   // Invert the rotation part 'r'. The inverse of a rotation matrix is
   // equal to its transpose.
   m[12] = m[13] = m[14] = 0;
   transpose(m);

   // inv(m) = inv(r) * inv(t)
   multiply(m, t);
}

static void
frustum(GLfloat *m,
        float left,
        float right,
        float bottom,
        float top,
        float nearval,
        float farval)
{
   float x, y, a, b, c, d;

   x = (2.0f * nearval) / (right - left);
   y = (2.0f * nearval) / (top - bottom);
   a = (right + left) / (right - left);
   b = (top + bottom) / (top - bottom);
   c = -(farval + nearval) / ( farval - nearval);
   d = -(2.0f * farval * nearval) / (farval - nearval);  /* error? */

#define M(row,col)  m[col*4+row]
   M (0,0) = x;     M (0,1) = 0.0f;  M (0,2) = a;      M (0,3) = 0.0f;
   M (1,0) = 0.0f;  M (1,1) = y;     M (1,2) = b;      M (1,3) = 0.0f;
   M (2,0) = 0.0f;  M (2,1) = 0.0f;  M (2,2) = c;      M (2,3) = d;
   M (3,0) = 0.0f;  M (3,1) = 0.0f;  M (3,2) = -1.0f;  M (3,3) = 0.0f;
#undef M
}

/**
 * Draws a gear.
 *
 * @param gear the gear to draw
 * @param transform the current transformation matrix
 * @param x the x position to draw the gear at
 * @param y the y position to draw the gear at
 * @param angle the rotation angle of the gear
 * @param color the color of the gear
 */
static void
draw_gear(struct gear *gear, GLfloat * transform,
          GLfloat x, GLfloat y, GLfloat angle, const GLfloat color[4])
{
   GLfloat model_view[16];
   GLfloat normal_matrix[16];
   GLfloat model_view_projection[16];

   /* Translate and rotate the gear */
   memcpy(model_view, transform, sizeof(model_view));
   translate(model_view, x, y, 0);
   rotate(model_view, 2 * M_PI * angle / 360.0, 0, 0, 1);

   /* Create and set the ModelViewProjectionMatrix */
   memcpy(model_view_projection, ProjectionMatrix,
          sizeof(model_view_projection));
   multiply(model_view_projection, model_view);

   glUniformMatrix4fv(ModelViewProjectionMatrix_location, 1, GL_FALSE,
                      model_view_projection);

   /*
    * Create and set the NormalMatrix. It's the inverse transpose of the
    * ModelView matrix.
    */
   memcpy(normal_matrix, model_view, sizeof(normal_matrix));
   invert(normal_matrix);
   transpose(normal_matrix);
   glUniformMatrix4fv(NormalMatrix_location, 1, GL_FALSE, normal_matrix);

   /* Set the gear color */
   glUniform4fv(MaterialColor_location, 1, color);

   /* Set the vertex buffer object to use */
   glBindBuffer(GL_ARRAY_BUFFER, gear->vbo);

   /* Set up the position of the attributes in the vertex buffer object */
   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                         6 * sizeof(GLfloat), NULL);
   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
                         6 * sizeof(GLfloat), (GLfloat *) 0 + 3);

   /* Enable the attributes */
   glEnableVertexAttribArray(0);
   glEnableVertexAttribArray(1);

   /* Draw the triangle strips that comprise the gear */
   int n;
   for (n = 0; n < gear->nstrips; n++)
      glDrawArrays(GL_TRIANGLE_STRIP, gear->strips[n].first,
                   gear->strips[n].count);

   /* Disable the attributes */
   glDisableVertexAttribArray(1);
   glDisableVertexAttribArray(0);
}

/**
 * Draws the gears.
 */
static void
gears_draw(const GLfloat *view_matrix)
{
   static const GLfloat red[4] = { 0.8, 0.1, 0.0, 1.0 };
   static const GLfloat green[4] = { 0.0, 0.8, 0.2, 1.0 };
   static const GLfloat blue[4] = { 0.2, 0.2, 1.0, 1.0 };
   GLfloat transform[16];

   memcpy(transform, view_matrix, sizeof(transform));

   /* Translate and rotate the view */
   translate(transform, 0, 0, -20);
   rotate(transform, 2 * M_PI * view_rot[0] / 360.0, 1, 0, 0);
   rotate(transform, 2 * M_PI * view_rot[1] / 360.0, 0, 1, 0);
   rotate(transform, 2 * M_PI * view_rot[2] / 360.0, 0, 0, 1);

   /* Draw the gears */
   draw_gear(gear1, transform, -3.0, -2.0, angle, red);
   draw_gear(gear2, transform, 3.1, -2.0, -2 * angle - 9.0, green);
   draw_gear(gear3, transform, -3.1, 4.2, -2 * angle - 25.0, blue);
}

static void
set_eye(struct stereo_renderer *renderer, int eye)
{
   if (eye == 0) {
      glViewport(0, renderer->layout.left_eye_y,
                 renderer->layout.eye_width,
                 renderer->layout.eye_height);
   } else {
      glViewport(renderer->layout.right_eye_x, 0,
                 renderer->layout.eye_width,
                 renderer->layout.eye_height);
   }
}

static void
redraw(struct stereo_renderer *renderer)
{
   GLfloat view_matrix[16];

   glClearColor(0.0, 0.0, 0.0, 1.0);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   /* First left eye.  */
   set_eye(renderer, 0);

   frustum(ProjectionMatrix, left, right, -asp, asp, 1.0, 1024.0);

   identity(view_matrix);
   translate(view_matrix, +0.5 * eyesep, 0.0, 0.0);
   gears_draw(view_matrix);

   /* Then right eye.  */
   set_eye(renderer, 1);

   frustum(ProjectionMatrix, -right, -left, -asp, asp, 1.0, 1024.0);

   identity(view_matrix);
   translate(view_matrix, -0.5 * eyesep, 0.0, 0.0);
   gears_draw(view_matrix);
}

/**
 * Handles a new window size or exposure.
 *
 * @param width the window width
 * @param height the window height
 */
static void
gears_reshape(int width, int height)
{
   GLfloat w;

   asp = (GLfloat) height / (GLfloat) width;
   w = fix_point * (1.0 / 5.0);

   left = -5.0 * ((w - 0.5 * eyesep) / fix_point);
   right = 5.0 * ((w + 0.5 * eyesep) / fix_point);
}

static int
get_elapsed_time(void)
{
   static int start_time = 0;
   int now;
   struct timeval tv;

   gettimeofday(&tv, NULL);

   now = tv.tv_sec * 1000 + tv.tv_usec / 1000;

   if (start_time == 0) {
      start_time = now;
      return 0;
   } else {
      return now - start_time;
   }
}

static void
gears_idle(void)
{
   static int frames = 0;
   static double tRot0 = -1.0, tRate0 = -1.0;
   double dt, t = get_elapsed_time() / 1000.0;

   if (tRot0 < 0.0)
      tRot0 = t;
   dt = t - tRot0;
   tRot0 = t;

   /* advance rotation for next frame */
   angle += 70.0 * dt;     /* 70 degrees per second */
   if (angle > 3600.0)
      angle -= 3600.0;

   view_rot[1] = angle / 2.0f;

   frames++;

   if (tRate0 < 0.0)
      tRate0 = t;
   if (t - tRate0 >= 5.0) {
      GLfloat seconds = t - tRate0;
      GLfloat fps = frames / seconds;
      printf("%d frames in %3.1f seconds = %6.3f FPS\n", frames,
             seconds, fps);
      tRate0 = t;
      frames = 0;
   }
}

static const char vertex_shader[] =
   "attribute vec3 position;\n"
   "attribute vec3 normal;\n"
   "\n"
   "uniform mat4 ModelViewProjectionMatrix;\n"
   "uniform mat4 NormalMatrix;\n"
   "uniform vec4 LightSourcePosition;\n"
   "uniform vec4 MaterialColor;\n"
   "\n"
   "varying vec4 Color;\n"
   "\n"
   "void main(void)\n"
   "{\n"
   "    // Transform the normal to eye coordinates\n"
   "    vec3 N = normalize(vec3(NormalMatrix * vec4(normal, 1.0)));\n"
   "\n"
   "    // The LightSourcePosition is actually its direction\n"
   "    // for directional light\n"
   "    vec3 L = normalize(LightSourcePosition.xyz);\n"
   "\n"
   "    // Multiply the diffuse value by the vertex color (which is\n"
   "    // fixed in this case) to get the actual color that we will\n"
   "    // use to draw this vertex with\n"
   "    float diffuse = max(dot(N, L), 0.0);\n"
   "    Color = vec4(diffuse * MaterialColor.rgb, 1.0);\n"
   "\n"
   "    // Transform the position to clip coordinates\n"
   "    gl_Position = ModelViewProjectionMatrix * vec4(position, 1.0);\n"
   "}";

static const char fragment_shader[] =
   "precision mediump float;\n"
   "varying vec4 Color;\n"
   "\n"
   "void main(void)\n"
   "{\n"
   "    gl_FragColor = Color;\n"
   "}";

static void
gears_init(void)
{
   GLuint v, f, program;
   const char *p;
   char msg[512];

   glEnable(GL_CULL_FACE);
   glEnable(GL_DEPTH_TEST);

   /* Compile the vertex shader */
   p = vertex_shader;
   v = glCreateShader(GL_VERTEX_SHADER);
   glShaderSource(v, 1, &p, NULL);
   glCompileShader(v);
   glGetShaderInfoLog(v, sizeof msg, NULL, msg);
   printf("vertex shader info: %s\n", msg);

   /* Compile the fragment shader */
   p = fragment_shader;
   f = glCreateShader(GL_FRAGMENT_SHADER);
   glShaderSource(f, 1, &p, NULL);
   glCompileShader(f);
   glGetShaderInfoLog(f, sizeof msg, NULL, msg);
   printf("fragment shader info: %s\n", msg);

   /* Create and link the shader program */
   program = glCreateProgram();
   glAttachShader(program, v);
   glAttachShader(program, f);
   glBindAttribLocation(program, 0, "position");
   glBindAttribLocation(program, 1, "normal");

   glLinkProgram(program);
   glGetProgramInfoLog(program, sizeof msg, NULL, msg);
   printf("info: %s\n", msg);

   /* Enable the shaders */
   glUseProgram(program);

   /* Get the locations of the uniforms so we can access them */
   ModelViewProjectionMatrix_location =
      glGetUniformLocation(program, "ModelViewProjectionMatrix");
   NormalMatrix_location = glGetUniformLocation(program, "NormalMatrix");
   LightSourcePosition_location =
      glGetUniformLocation(program, "LightSourcePosition");
   MaterialColor_location = glGetUniformLocation(program, "MaterialColor");

   /* Set the LightSourcePosition uniform which is constant
    * throught the program */
   glUniform4fv(LightSourcePosition_location, 1, LightSourcePosition);

   /* make the gears */
   gear1 = create_gear(1.0, 4.0, 1.0, 20, 0.7);
   gear2 = create_gear(0.5, 2.0, 2.0, 10, 0.7);
   gear3 = create_gear(1.3, 2.0, 0.5, 10, 0.7);
}

static void
draw(struct stereo_renderer *renderer)
{
   gears_idle();
   redraw(renderer);
}

static struct stereo_renderer *
create_renderer(const struct mode_layout *layout)
{
   struct stereo_renderer *renderer;

   renderer = xmalloc(sizeof *renderer);
   memset(renderer, 0, sizeof *renderer);

   renderer->layout = *layout;

   gears_init();
   gears_reshape(layout->virtual_eye_width, layout->virtual_eye_height);

   return renderer;
}

static void
renderer_free(struct stereo_renderer *renderer)
{
   free(renderer);
}

static void
sigint_handler(int sig)
{
   UNUSED(sig);

   quit = 1;
}

static void
main_loop(struct stereo_data *data)
{
   struct sigaction action = {
      .sa_handler = sigint_handler,
   };
   struct sigaction old_action;

   sigemptyset(&action.sa_mask);
   sigaction(SIGINT, &action, &old_action);

   while (!quit) {
      draw(data->renderer);
      swap(data->winsys);
   }

   sigaction(SIGINT, &old_action, NULL);
}

static void
usage(void)
{
   printf("usage: stereo-es2gears [OPTION]...\n"
          "\n"
          "  -h              Show this help message\n"
          "  -c <connector>  Set a connector to display on\n"
          "  -d <device>     Set the DRI device to open\n"
          "  -l <layout>     Stereo layout (none/fp/sbsf/tb/sbsh)\n");
   exit(0);
}

static int
process_options(struct stereo_options *options, int argc, char **argv)
{
   static const char args[] = "-c:l:h";
   int opt;

   memset(options, 0, sizeof *options);

   options->connector = 0;

   while ((opt = getopt(argc, argv, args)) != -1) {
      switch (opt) {
      case 'h':
         usage();
         break;
      case 'd':
         options->card = optarg;
         break;
      case 'c':
         options->connector = atoi(optarg);
         break;
      case 'l':
         options->stereo_layout = optarg;
         break;

      case ':':
      case '?':
         return EXIT_FAILURE;

      case '\1':
         fprintf(stderr, "unexpected argument \"%s\"\n", optarg);
         return EXIT_FAILURE;
      default:
         fprintf(stderr, "unexpected argument \"-%c\"\n", opt);
         return EXIT_FAILURE;
      }
   }

   return 0;
}

int
main(int argc, char **argv)
{
   int ret = EXIT_SUCCESS;
   struct stereo_data data;
   struct stereo_options options;

   memset(&data, 0, sizeof data);

   ret = process_options(&options, argc, argv);
   if (ret)
      goto out;

   data.winsys = create_winsys(&options);
   if (data.winsys == NULL) {
      ret = EXIT_FAILURE;
      goto out;
   }

   data.renderer = create_renderer(&data.winsys->dev->layout);
   if (data.renderer == NULL) {
      ret = EXIT_FAILURE;
      goto out;
   }

   main_loop(&data);

out:
   /* cleanup everything */
   if (data.renderer)
      renderer_free(data.renderer);
   if (data.winsys)
      winsys_free(data.winsys);

   return ret;
}
