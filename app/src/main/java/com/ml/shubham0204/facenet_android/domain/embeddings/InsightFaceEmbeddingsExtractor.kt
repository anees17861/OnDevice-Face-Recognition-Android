package com.ml.shubham0204.facenet_android.domain.embeddings

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.koin.core.annotation.Single
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.IOException
import java.nio.FloatBuffer
import kotlin.math.sqrt

@Single
class InsightFaceEmbeddingsExtractor(private val context: Context) {

    companion object {
        private const val TAG = "InsightFaceExtractor"
        private const val MODEL_NAME = "models/w600k_r50.tflite" // Place your model in assets folder
        private const val INPUT_SIZE = 112 // Buffalo Large typically uses 112x112 input
        private const val EMBEDDING_SIZE = 512 // Buffalo Large produces 512-dimensional embeddings
        private const val CHANNELS = 3
    }

    private var interpreter: Interpreter? = null
    private var isModelLoaded = false

    init {
        loadModel()
    }

    /**
     * Load the TensorFlow Lite model from assets
     */
    private fun loadModel() {
        try {
            val modelBuffer = FileUtil.loadMappedFile(context, MODEL_NAME)
            val options = Interpreter.Options().apply {
                setNumThreads(4) // Adjust based on device capabilities
                setUseNNAPI(true) // Use Android Neural Networks API if available
            }
            interpreter = Interpreter(modelBuffer, options)
            isModelLoaded = true
            Log.d(TAG, "Model loaded successfully")
        } catch (e: IOException) {
            Log.e(TAG, "Failed to load model: ${e.message}")
            isModelLoaded = false
        }
    }

    /**
     * Extract face embedding from a cropped face bitmap
     * @param croppedFace Bitmap containing a cropped face
     * @return FloatArray containing the face embedding or null if extraction fails
     */
    fun extractEmbedding(croppedFace: Bitmap): FloatArray? {
        if (!isModelLoaded || interpreter == null) {
            Log.e(TAG, "Model not loaded")
            return null
        }

        return try {
            // Preprocess the image
            val preprocessedImage = preprocessImage(croppedFace)

            // Prepare input and output buffers
//            val inputBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * CHANNELS)
//            inputBuffer.order(ByteOrder.nativeOrder())

            val outputBuffer = Array(1) { FloatArray(EMBEDDING_SIZE) }

//            // Fill input buffer with preprocessed image data
//            preprocessedImage.forEach { pixel ->
//                inputBuffer.putFloat(pixel)
//            }

            // Run inference
            interpreter?.run(preprocessedImage, outputBuffer)

            // Normalize the embedding (L2 normalization)
            val embedding = outputBuffer[0]
            normalizeVector(embedding)

            Log.d(TAG, "Embedding extracted successfully, size: ${embedding.size}")
            embedding

        } catch (e: Exception) {
            Log.e(TAG, "Error during embedding extraction: ${e.message}")
            null
        }
    }

    /**
     * Preprocess the input bitmap for the model
     * @param bitmap Input bitmap
     * @return FloatArray with normalized pixel values
     */
    private fun preprocessImage(bitmap: Bitmap): FloatBuffer {
        // Resize bitmap to model input size
        val resizedBitmap = if (bitmap.width != INPUT_SIZE || bitmap.height != INPUT_SIZE) {
            Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        } else {
            bitmap
        }
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resizedBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

//        val normalizedPixels = FloatArray(CHANNELS*INPUT_SIZE * INPUT_SIZE )

        // Convert RGB pixels to normalized float values
        // InsightFace models typically expect RGB values normalized to [-1, 1] or [0, 1]
        // Adjust normalization based on your specific model requirements
//        for (i in pixels.indices) {
//            val pixel = pixels[i]
//            val r = (pixel shr 16 and 0xFF) -127.5f/ 128f  // Normalize to [-1, 1]
//            val g = (pixel shr 8 and 0xFF) -127.5f / 128f
//            val b = (pixel and 0xFF) -127.5f / 128f
//
//            // RGB format for most InsightFace models
//            normalizedPixels[i * 3] = r
//            normalizedPixels[i * 3 + 1] = g
//            normalizedPixels[i * 3 + 2] = b
//        }
//
//        return normalizedPixels

        val inputBuffer = FloatBuffer.allocate(3 * INPUT_SIZE * INPUT_SIZE)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF)
            val g = ((pixel shr 8) and 0xFF)
            val b = (pixel and 0xFF)

            // Normalize to [-1, 1] range (common for InsightFace models)
            inputBuffer.put((r - 127.5f)/128.0f) // R channel
        }

        for (i in pixels.indices) {
            val pixel = pixels[i]
            val g = ((pixel shr 8) and 0xFF)
            inputBuffer.put((g - 127.5f)/128.0f) // G channel
        }

        for (i in pixels.indices) {
            val pixel = pixels[i]
            val b = (pixel and 0xFF)
            inputBuffer.put((b - 127.5f)/128.0f) // B channel
        }

        inputBuffer.rewind()
        val shape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        return inputBuffer
    }

    /**
     * Apply L2 normalization to the embedding vector
     * @param vector Input vector to normalize
     */
    private fun normalizeVector(vector: FloatArray) {
        var norm = 0.0f
        for (value in vector) {
            norm += value * value
        }
        norm = sqrt(norm)

        if (norm > 0) {
            for (i in vector.indices) {
                vector[i] /= norm
            }
        }
    }

    /**
     * Calculate cosine similarity between two embeddings
     * @param embedding1 First embedding
     * @param embedding2 Second embedding
     * @return Similarity score between -1 and 1
     */
    fun calculateSimilarity(embedding1: FloatArray, embedding2: FloatArray): Float {
        if (embedding1.size != embedding2.size) {
            throw IllegalArgumentException("Embeddings must have the same size")
        }

        var dotProduct = 0.0f
        for (i in embedding1.indices) {
            dotProduct += embedding1[i] * embedding2[i]
        }

        return dotProduct // Since vectors are already normalized, dot product = cosine similarity
    }

    /**
     * Check if two faces are the same person based on similarity threshold
     * @param embedding1 First face embedding
     * @param embedding2 Second face embedding
     * @param threshold Similarity threshold (default: 0.6)
     * @return Boolean indicating if faces match
     */
    fun areSamePerson(
        embedding1: FloatArray,
        embedding2: FloatArray,
        threshold: Float = 0.6f
    ): Boolean {
        val similarity = calculateSimilarity(embedding1, embedding2)
        return similarity >= threshold
    }

    /**
     * Clean up resources
     */
    fun close() {
        interpreter?.close()
        interpreter = null
        isModelLoaded = false
        Log.d(TAG, "Resources cleaned up")
    }

    /**
     * Check if model is ready for inference
     */
    fun isReady(): Boolean = isModelLoaded && interpreter != null
}
