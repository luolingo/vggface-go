package vfworkers

import (
	"mface/vggface"

	"676f.dev/utilities/structures/stack"
	"github.com/gammazero/workerpool"
)

const CAFFE_PROTOTXT_FILEPATH = "/root/face/vgg_face/vgg_face_caffe/VGG_FACE_deploy.prototxt"
const CAFFE_MODEL_FILEPATH = "/root/face/vgg_face/vgg_face_caffe/VGG_FACE.caffemodel"

type Workers struct {
	vfwkers *stack.Stack
	wp      *workerpool.WorkerPool
}

func New() Workers {
	return Workers{
		vfwkers: stack.NewStack(),
		wp:      workerpool.New(5),
	}
}

func (me *Workers) Release() {
	me.wp.Stop()
	for me.vfwkers.Len() > 0 {
		vgf := me.vfwkers.Pop().(vggface.VggFace)
		vgf.Relase()
	}
}

func (me *Workers) MatchingFace(fa []byte, fb []byte) (float32, error) {
	var ret float32
	var err error

	me.wp.SubmitWait(func() {
		if me.vfwkers.Len() <= 0 {
			me.vfwkers.Push(vggface.New(CAFFE_PROTOTXT_FILEPATH, CAFFE_MODEL_FILEPATH))
		}

		vgf := me.vfwkers.Pop().(vggface.VggFace)
		ret, err = vgf.Matching(fa, fb)
		me.vfwkers.Push(vgf)
	})

	return ret, err
}
