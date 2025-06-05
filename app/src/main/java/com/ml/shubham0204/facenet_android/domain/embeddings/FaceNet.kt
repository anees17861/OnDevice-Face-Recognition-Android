package com.ml.shubham0204.facenet_android.domain.embeddings

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.nio.ByteBuffer
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.koin.core.annotation.Single
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorOperator
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat
import java.nio.ByteOrder

// Derived from the original project:
// https://github.com/shubham0204/FaceRecognition_With_FaceNet_Android/blob/master/app/src/main/java/com/ml/quaterion/facenetdetection/model/FaceNetModel.kt
// Utility class for FaceNet model
@Single
class FaceNet(context: Context, useGpu: Boolean = true, useXNNPack: Boolean = true) {

    // Input image size for FaceNet model.
//    private val imgSize = 160
    private val imgSize = 112

    // Output embedding size
    private val embeddingDim = 512

    private var interpreter: Interpreter
    private val imageTensorProcessor =
        ImageProcessor.Builder()
            .add(ResizeOp(imgSize, imgSize, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp())
//            .add(StandardizeOp())
            .build()

    init {
        // Initialize TFLiteInterpreter
        val interpreterOptions =
            Interpreter.Options().apply {
                // Add the GPU Delegate if supported.
                // See -> https://www.tensorflow.org/lite/performance/gpu#android
                if (useGpu) {
                    if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                        addDelegate(GpuDelegate(CompatibilityList().bestOptionsForThisDevice))
                    }
                } else {
                    // Number of threads for computation
                    numThreads = 4
                }
                useXNNPACK = useXNNPack
                useNNAPI = true
                // Configure NNAPI delegate with options
//                val nnApiOptions = NnApiDelegate.Options().apply {
//                    setExecutionPreference(2)
//                    // Use allowFp16 = false for better precision at the cost of speed
//                    allowFp16 = false
//                    // Set the acceleration mode
//                    acceleratorName = null  // null means use any available accelerator
//                    useNNAPI = false
//                }
//
//                addDelegate(NnApiDelegate(nnApiOptions))
            }
        interpreter =
            Interpreter(FileUtil.loadMappedFile(context, "models/w600k_r50_float32.tflite"), interpreterOptions)
    }

    // Gets an face embedding using FaceNet
    suspend fun getFaceEmbedding(image: Bitmap) =
        withContext(Dispatchers.Default) {
            val embeddings = runFaceNet(convertBitmapToBuffer(image))[0]
            val embeddings_norm = l2Normalize(embeddings)
//            Log.d("FaceNet", embeddings_norm.max().toString() + " " + embeddings_norm.min().toString())
            return@withContext  embeddings_norm// Add L2 normalization

//            return@withContext runFaceNet(convertBitmapToBuffer(image))[0]
        }

    // Run the FaceNet model
    private fun runFaceNet(inputs: Any): Array<FloatArray> {
        val faceNetModelOutputs = Array(1) { FloatArray(embeddingDim) }
//        Log.d("FaceNet", "faceNetModelOutputs: $faceNetModelOutputs")
//        Log.d("FaceNet", "inputs: $inputs")
        interpreter.run(inputs, faceNetModelOutputs)
        return faceNetModelOutputs
    }

//    // Resize the given bitmap and convert it to a ByteBuffer
    private fun convertBitmapToBuffer(image: Bitmap): ByteBuffer {
        return imageTensorProcessor.process(TensorImage.fromBitmap(image)).buffer
    }

//    private fun convertBitmapToBuffer(image: Bitmap): ByteBuffer {
//        // Convert the image to TensorImage and process it
//        val tensorImage = imageTensorProcessor.process(TensorImage.fromBitmap(image))
//        val rgbPixels = tensorImage.tensorBuffer.floatArray
//
//        // Prepare a ByteBuffer for NCHW format
//        val buffer = ByteBuffer.allocateDirect(4 * imgSize * imgSize * 3) // 4 bytes per float
//        buffer.order(ByteOrder.nativeOrder())
//
//        // Rearrange the data to NCHW format
//        val channelSize = imgSize * imgSize
//        for (c in 0 until 3) { // Iterate over channels (R, G, B)
//            for (i in 0 until channelSize) {
//                buffer.putFloat(rgbPixels[i * 3 + c]) // Extract channel-first data
//            }
//        }
//
//        buffer.rewind()
//        return buffer
//    }

    // Op to perform standardization
    // x' = ( x - mean ) / std_dev
    class StandardizeOp : TensorOperator {

        override fun apply(p0: TensorBuffer?): TensorBuffer {
            val pixels = p0!!.floatArray
            val mean = pixels.average().toFloat()
            var std = sqrt(pixels.map { pi -> (pi - mean).pow(2) }.sum() / pixels.size.toFloat())
            std = max(std, 1f / sqrt(pixels.size.toFloat()))
            for (i in pixels.indices) {
                pixels[i] = (pixels[i] - mean) / std
            }
            val output = TensorBufferFloat.createFixedSize(p0.shape, DataType.FLOAT32)
            output.loadArray(pixels)
            return output
        }
    }

    // L2 normalization implementation
    private fun l2Normalize(embeddings: FloatArray): FloatArray {
        val norm = sqrt(embeddings.map { it * it }.sum())
        return if (norm > 0) {
            embeddings.map { it / norm }.toFloatArray()
        } else {
            embeddings
        }
    }

    class NormalizeOp : TensorOperator {
        override fun apply(p0: TensorBuffer?): TensorBuffer {
            val pixels = p0!!.floatArray

            // InsightFace normalization: (x - 127.5) / 128.0
            for (i in pixels.indices) {
                pixels[i] = (pixels[i] - 127.5f) / 128.0f
            }

            val output = TensorBufferFloat.createFixedSize(p0.shape, DataType.FLOAT32)
            output.loadArray(pixels)
            return output
        }
    }



}
