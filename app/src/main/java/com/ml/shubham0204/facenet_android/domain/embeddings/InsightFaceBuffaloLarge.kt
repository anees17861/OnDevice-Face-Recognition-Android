package com.ml.shubham0204.facenet_android.domain.embeddings


import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.*
import ai.onnxruntime.providers.NNAPIFlags
import org.koin.core.annotation.Single
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.util.EnumSet
import kotlin.math.sqrt

@Single
class InsightFaceBuffaloLarge(private val context: Context) {

    private var ortSession: OrtSession? = null
    private var ortEnvironment: OrtEnvironment? = null

    companion object {
        private const val TAG = "InsightFaceBuffalo"
        private const val MODEL_NAME = "w600k_r50.onnx" // Place this in assets folder
        private const val INPUT_SIZE = 112 // Standard input size for Buffalo Large
        private const val EMBEDDING_SIZE = 512 // Output embedding dimension
    }
    fun loadOnnxModelFromAssets(context: Context, modelName: String): OrtSession {
        val assetManager = context.assets
        val inputStream = assetManager.open(modelName)
        val outFile = File(context.cacheDir, modelName)

        inputStream.use { input ->
            FileOutputStream(outFile).use { output ->
                input.copyTo(output)
            }
        }

        val env = OrtEnvironment.getEnvironment()
//        val sessionOptions = OrtSession.SessionOptions()
        val sessionOptions = OrtSession.SessionOptions().apply {
            // Use CPU provider (you can change to GPU if available)
//            addConfigEntry("session.load_model_format", "ORT")
        }
        sessionOptions.addNnapi()
//        sessionOptions.addNnapi(EnumSet.of(NNAPIFlags.USE_NCHW, NNAPIFlags.CPU_DISABLED))
        return env.createSession(outFile.absolutePath, sessionOptions)
    }
    /**
     * Initialize the ONNX model
     */
    fun initialize(): Boolean {
        return try {
            ortEnvironment = OrtEnvironment.getEnvironment()

            // Load model from assets
//            val modelBytes = context.assets.open(MODEL_NAME).readBytes()
//            val modelInputStream = context.assets.open("w600k_r50.onnx")
//            context.assets.

            // Create session options
//            val sessionOptions = OrtSession.SessionOptions().apply {
//                // Use CPU provider (you can change to GPU if available)
//                addConfigEntry("session.load_model_format", "ORT")
//                addNnapi()
//            }

            // Create session
//            ortSession = ortEnvironment?.createSession(MODEL_NAME, sessionOptions)
            ortSession = loadOnnxModelFromAssets(context, "w600k_r50.onnx")

            Log.d(TAG, "InsightFace Buffalo Large model initialized successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize model", e)
            false
        }
    }

    /**
     * Extract face embedding from a cropped and aligned face image
     * @param croppedFace Bitmap of cropped and aligned face (should be 112x112)
     * @return FloatArray containing the face embedding (512 dimensions) or null if failed
     */
    fun extractEmbedding(croppedFace: Bitmap): FloatArray? {
        return ortSession?.let { session ->
                // Preprocess the image
                val inputTensor = preprocessImage(croppedFace)
                Log.d(TAG, "Preprocessed the image")
                // Run inference
                val inputs = mapOf("input.1" to inputTensor)
                val outputs = session.run(inputs)
                Log.d(TAG, "Run INFERENECE")

                // Extract embedding from output
                val outputTensor = outputs[0].value as Array<FloatArray>
                val embedding = outputTensor[0]
                Log.d(TAG, "Extract embedding from output")
                // Normalize the embedding (L2 normalization)
                normalizeEmbedding(embedding)
                Log.d(TAG, "Normalized the EMBeddings")

                // Clean up
                inputTensor.close()
//                outputs.forEach { it.close() }

                Log.d(TAG, "Embedding extracted successfully, size: ${embedding.size}")
                return embedding
            }
//        } catch (e: Exception) {
//            Log.e(TAG, "Failed to extract embedding", e)
//            null
//        }
    }

    /**
     * Preprocess the input image for the model
     * - Resize to 112x112 if needed
     * - Normalize pixel values to [-1, 1]
     * - Convert to CHW format (Channels, Height, Width)
     */
    private fun preprocessImage(bitmap: Bitmap): OnnxTensor {
        // Resize bitmap if necessary
        val resizedBitmap = if (bitmap.width != INPUT_SIZE || bitmap.height != INPUT_SIZE) {
            Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        } else {
            bitmap
        }

        // Convert bitmap to float array in CHW format
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resizedBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        // Create float buffer for RGB channels separately (CHW format)
        val inputBuffer = FloatBuffer.allocate(3 * INPUT_SIZE * INPUT_SIZE)

        // Extract and normalize RGB channels
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

        // Create tensor with shape [1, 3, 112, 112] (NCHW format)
        val shape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        return OnnxTensor.createTensor(ortEnvironment, inputBuffer, shape)
    }

    /**
     * Apply L2 normalization to the embedding vector
     */
    private fun normalizeEmbedding(embedding: FloatArray): FloatArray {
        val norm = sqrt(embedding.map { it * it }.sum())
        if (norm > 0) {
            for (i in embedding.indices) {
                embedding[i] = embedding[i] / norm
            }
        }
        return embedding
    }

    /**
     * Calculate cosine similarity between two embeddings
     */
    fun cosineSimilarity(embedding1: FloatArray, embedding2: FloatArray): Float {
        if (embedding1.size != embedding2.size) {
            throw IllegalArgumentException("Embeddings must have the same size")
        }

        var dotProduct = 0.0f
        var norm1 = 0.0f
        var norm2 = 0.0f

        for (i in embedding1.indices) {
            dotProduct += embedding1[i] * embedding2[i]
            norm1 += embedding1[i] * embedding1[i]
            norm2 += embedding2[i] * embedding2[i]
        }

        return dotProduct / (sqrt(norm1) * sqrt(norm2))
    }

    /**
     * Calculate Euclidean distance between two embeddings
     */
    fun euclideanDistance(embedding1: FloatArray, embedding2: FloatArray): Float {
        if (embedding1.size != embedding2.size) {
            throw IllegalArgumentException("Embeddings must have the same size")
        }

        var sum = 0.0f
        for (i in embedding1.indices) {
            val diff = embedding1[i] - embedding2[i]
            sum += diff * diff
        }

        return sqrt(sum)
    }

    /**
     * Clean up resources
     */
    fun cleanup() {
        try {
            ortSession?.close()
            ortEnvironment?.close()
            Log.d(TAG, "Resources cleaned up successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error during cleanup", e)
        }
    }
}