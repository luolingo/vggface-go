package main

import "C"
import (
	"bufio"
	"fmt"
	"mface/vfworkers"
	"os"
)

func readimg(fp string) []byte {
	file, err := os.Open(fp)
	if err != nil {
		fmt.Println(err)
		return nil
	}
	defer file.Close()

	stat, err := file.Stat()
	if err != nil {
		fmt.Println(err)
		return nil
	}

	bs := make([]byte, stat.Size())
	bufio.NewReader(file).Read(bs)
	return bs
}

func main() {
	wfers := vfworkers.New()
	bufa := readimg("/root/face/vgg_face/IMG_00101.jpg")
	if bufa == nil {
		fmt.Println("read imageA file error")
		return
	}
	bufb := readimg("/root/face/vgg_face/3.jpg")
	if bufa == nil {
		fmt.Println("read imageB file error")
		return
	}
	sim, err := wfers.MatchingFace(bufa, bufb)
	wfers.Release()
	fmt.Printf("sim = %v, err = %v", sim, err)
}
