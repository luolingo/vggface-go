package vggface

// #include <stdio.h>
// #include <stdlib.h>
// #cgo LDFLAGS: -L/root/face/vgg_face/src -lClassifier -lglog  -lcaffe  -L/usr/local/lib -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn_objdetect -lopencv_dpm -lopencv_highgui -lopencv_videoio -lopencv_face -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_img_hash -lopencv_line_descriptor -lopencv_optflow -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ml -lopencv_ximgproc -lopencv_xobjdetect -lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_flann -lopencv_xphoto -lopencv_photo -lopencv_imgproc -lopencv_core
// void* ClassifierInit(const char *model_file, const char *trained_file);
// float ClassifierMatching(void *pClassifier, int rowsa, int colsa, char* imga, int rowsb, int colsb, char* imgb);
// void ClassifierRelease(void *pClassifier);
import "C"
import (
	"bytes"
	"errors"
	"fmt"
	"image"
	"image/draw"
	"image/jpeg"
	"math"
	"unsafe"

	resizedraw "golang.org/x/image/draw"
)

const IMG_MODEL_WIDTH = 224.0
const IMG_MODEL_HEIGHT = 224.0

func imageResize(src image.Image) (image.Image, error) {
	if src.Bounds().Max.X == IMG_MODEL_WIDTH && src.Bounds().Max.Y == IMG_MODEL_HEIGHT {
		return src, nil
	}

	if math.Abs(float64(src.Bounds().Max.X)/float64(src.Bounds().Max.Y)-IMG_MODEL_WIDTH/IMG_MODEL_HEIGHT) > 0.0001 {
		return nil, errors.New("invalid image size")
	}

	dst := image.NewRGBA(image.Rect(0, 0, IMG_MODEL_WIDTH, IMG_MODEL_HEIGHT))
	resizedraw.NearestNeighbor.Scale(dst, dst.Rect, src, src.Bounds(), draw.Over, nil)
	return dst, nil
}

func imageToRGBA(src image.Image) *image.RGBA {
	if dst, ok := src.(*image.RGBA); ok {
		return dst
	}

	b := src.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(dst, dst.Bounds(), src, b.Min, draw.Src)
	return dst
}

func imageToRGB(tmp image.Image) []uint8 {
	img := imageToRGBA(tmp)

	sz := img.Bounds()
	raw := make([]uint8, (sz.Max.X-sz.Min.X)*(sz.Max.Y-sz.Min.Y)*3)
	idx := 0
	for y := sz.Min.Y; y < sz.Max.Y; y++ {
		for x := sz.Min.X; x < sz.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			raw[idx], raw[idx+1], raw[idx+2] = uint8(b), uint8(g), uint8(r)
			idx += 3
		}
	}
	return raw
}

type VggFace struct {
	Classifier unsafe.Pointer
}

func New(prototxtFpath string, caffemodelFpath string) VggFace {
	prototxt := C.CString(prototxtFpath)
	caffemodel := C.CString(caffemodelFpath)
	defer C.free(unsafe.Pointer(prototxt))
	defer C.free(unsafe.Pointer(caffemodel))

	classifier := C.ClassifierInit(prototxt, caffemodel)
	return VggFace{
		classifier,
	}
}

func (me *VggFace) Relase() {
	C.ClassifierRelease(me.Classifier)
}

func (me *VggFace) Matching(imgfa []byte, imgfb []byte) (float32, error) {
	imga, err := jpeg.Decode(bytes.NewReader(imgfa))
	if err != nil {
		return 0.0, errors.New(fmt.Sprintf("read image a failed: %v", err))
	}

	imga, err = imageResize(imga)
	if err != nil {
		return 0.0, errors.New(fmt.Sprintf("resize image a failed: %v", err))
	}

	bufa := imageToRGB(imga)

	imgb, err := jpeg.Decode(bytes.NewReader(imgfb))
	if err != nil {
		return 0.0, errors.New(fmt.Sprintf("read image b failed: %v", err))
	}

	imgb, err = imageResize(imgb)
	if err != nil {
		return 0.0, errors.New(fmt.Sprintf("resize image b failed: %v", err))
	}

	bufb := imageToRGB(imgb)

	sim := C.ClassifierMatching(me.Classifier, C.int(imga.Bounds().Max.X), C.int(imga.Bounds().Max.Y), (*C.char)(unsafe.Pointer(&bufa[0])),
		C.int(imgb.Bounds().Max.X), C.int(imgb.Bounds().Max.Y), (*C.char)(unsafe.Pointer(&bufb[0])))

	return float32(sim), nil
}
