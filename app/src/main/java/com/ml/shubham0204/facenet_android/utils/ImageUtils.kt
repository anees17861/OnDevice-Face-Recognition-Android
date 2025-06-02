package com.ml.shubham0204.facenet_android.utils

import android.graphics.ImageFormat
import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer


object ImageUtils {
    fun convertYUVToARGB(image: ImageProxy, out: IntArray) {
        if (image.format != ImageFormat.YUV_420_888) {
            throw IllegalArgumentException("Unsupported image format");
        }

        val width = image.width
        val height = image.height

        val planes: Array<ImageProxy.PlaneProxy> = image.planes

        val yBuffer: ByteBuffer = planes[0].buffer
        val uBuffer: ByteBuffer = planes[1].buffer
        val vBuffer: ByteBuffer = planes[2].buffer

        val yRowStride: Int = planes[0].rowStride
        val uvRowStride: Int = planes[1].rowStride
        val uvPixelStride: Int = planes[1].pixelStride


        val yBytes = ByteArray(yBuffer.remaining())
        val uBytes = ByteArray(uBuffer.remaining())
        val vBytes = ByteArray(vBuffer.remaining())

        yBuffer[yBytes]
        uBuffer[uBytes]
        vBuffer[vBytes]

        var yp = 0
        for (j in 0..<height) {
            val pY = j * yRowStride
            val pUV = (j shr 1) * uvRowStride

            for (i in 0..<width) {
                val uvOffset = pUV + (i shr 1) * uvPixelStride

                val y = 0xff and yBytes[pY + i].toInt()
                val u = 0xff and uBytes[uvOffset].toInt()
                val v = 0xff and vBytes[uvOffset].toInt()

                out[yp] = yuvToRgbPixel(y, u, v)
                yp++
            }
        }
    }

    private fun yuvToRgbPixel(y: Int, u: Int, v: Int): Int {
        // Adjust and check YUV values
        var y = y
        var u = u
        var v = v
        y = if ((y - 16) < 0) 0 else (y - 16)
        u -= 128
        v -= 128

        // This is the floating point equivalent. We do the conversion in integer
        // because some Android devices do not have floating point in hardware.
        // nR = (int)(1.164 * nY + 2.018 * nU);
        // nG = (int)(1.164 * nY - 0.813 * nV - 0.391 * nU);
        // nB = (int)(1.164 * nY + 1.596 * nV);
        val y1192 = 1192 * y
        var r = (y1192 + 1634 * v)
        var g = (y1192 - 833 * v - 400 * u)
        var b = (y1192 + 2066 * u)

        // Clipping RGB values to be inside boundaries [ 0 , kMaxChannelValue ]
        val kMaxChannelValue = 0x3FFFF // 2^18 - 1
        r = r.coerceIn(0, kMaxChannelValue)
        g = g.coerceIn(0, kMaxChannelValue)
        b = b.coerceIn(0, kMaxChannelValue)

        // Each color is 18 bits wide, so we need to downscale and thus strip to most significant 8 bits
        // b shifted right by 18 - (8*1) = 10
        // g shifted right by 18 - (8*2) = 2
        // r shifted right by 18 - (8*3) = -6 => shift left +6
        // mask is applied to strip off the rest 10 bits

        val aMask = 0xFF000000.toInt()
        val rMask = 0x00FF0000
        val gMask = 0x0000FF00
        val bMask = 0x000000FF

        return aMask or
                ((r shl 6) and rMask) or
                ((g shr 2) and gMask) or
                ((b shr 10) and bMask)
    }

}