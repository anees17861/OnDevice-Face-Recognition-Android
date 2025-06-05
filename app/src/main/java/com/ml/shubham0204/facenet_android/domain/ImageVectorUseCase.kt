package com.ml.shubham0204.facenet_android.domain

import android.graphics.Bitmap
import android.graphics.Rect
import android.net.Uri
import android.util.Log
import com.ml.shubham0204.facenet_android.data.FaceImageRecord
import com.ml.shubham0204.facenet_android.data.IdentityAggregatorRepository
import com.ml.shubham0204.facenet_android.data.ImagesVectorDB
import com.ml.shubham0204.facenet_android.data.PersonDB
import com.ml.shubham0204.facenet_android.data.RecognitionMetrics
import com.ml.shubham0204.facenet_android.data.ThresholdPreferenceRepository
import com.ml.shubham0204.facenet_android.domain.embeddings.FaceNet
import com.ml.shubham0204.facenet_android.domain.face_detection.FaceSpoofDetector
import com.ml.shubham0204.facenet_android.domain.face_detection.MLKitFaceDetector
import com.ml.shubham0204.facenet_android.domain.face_detection.MediapipeFaceDetector
import com.ml.shubham0204.facenet_android.util.BatchedFileLogger
import org.koin.core.annotation.Single
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.imgproc.Imgproc
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.time.DurationUnit
import kotlin.time.measureTimedValue
//import com.ml.shubham0204.facenet_android.domain.embeddings.InsightFaceBuffaloLarge
//import com.ml.shubham0204.facenet_android.domain.embeddings.InsightFaceEmbeddingsExtractor

@Single
class ImageVectorUseCase(
//    private val mlKitFaceDetector: MLKitFaceDetector,
    private val mediapipeFaceDetector: MediapipeFaceDetector,
    private val faceSpoofDetector: FaceSpoofDetector,
    private val imagesVectorDB: ImagesVectorDB,
    private val personDB: PersonDB,
    private val faceNet: FaceNet,
//    private val insightFaceBuffaloLarge: InsightFaceBuffaloLarge,
//    private  val insightFaceEmbeddingsExtractor: InsightFaceEmbeddingsExtractor,

    private val thresholdRepo: ThresholdPreferenceRepository,
    private val identityAggregatorRepository: IdentityAggregatorRepository


) {

    data class FaceRecognitionResult(
        val personName: String,
        val boundingBox: Rect,
        val spoofResult: FaceSpoofDetector.FaceSpoofResult? = null
    )
    fun l2Norm(v: FloatArray): Double {
        var sum = 0.0
        for (x in v) sum += x*x
        return sqrt(sum)
    }
    // Add the person's image to the database
    suspend fun addImage(personID: Long, personName: String, imageUri: Uri): Result<Boolean> {
        // Perform face-detection and get the cropped face as a Bitmap
        val faceDetectionResult = mediapipeFaceDetector.getCroppedFace(imageUri)
//        val faceDetectionResult = mlKitFaceDetector.getCroppedFace(imageUri)
        if (faceDetectionResult.isSuccess) {
            // Get the embedding for the cropped face, and store it
            // in the database, along with `personId` and `personName`
            Log.d("ImageVectorUseCase", "Adding image for person: $personName")
            val embedding = faceNet.getFaceEmbedding(faceDetectionResult.getOrNull()!!)
//            insightFaceEmbeddingsExtractor.isReady()
//            val embedding = insightFaceEmbeddingsExtractor.extractEmbedding(faceDetectionResult.getOrNull()!!)
//            insightFaceBuffaloLarge.initialize()
//            val embedding = insightFaceBuffaloLarge.extractEmbedding(faceDetectionResult.getOrNull()!!)
//            insightFaceBuffaloLarge.cleanup()
//            Log.d("ImageVectorUseCase", "GOT THE EMBEDDING")
            imagesVectorDB.addFaceImageRecord(
                FaceImageRecord(
                    personID = personID,
                    personName = personName,
                    faceEmbedding = embedding!!
                )
            )
            return Result.success(true)
        } else {
            Log.e("addImage",faceDetectionResult.exceptionOrNull().toString())
            return Result.failure(faceDetectionResult.exceptionOrNull()!!)
        }
    }

    // From the given frame, return the name of the person by performing
    // face recognition
    suspend fun getNearestPersonName(
        frameBitmap: Bitmap
    ): Pair<RecognitionMetrics?, List<FaceRecognitionResult>> {
        // Perform face-detection and get the cropped face as a Bitmap
        val (faceDetectionResult, t1) =
            measureTimedValue { mediapipeFaceDetector.getAllCroppedFaces(frameBitmap) }
//        val (faceDetectionResult, t1) =
//            measureTimedValue { mlKitFaceDetector.getAllCroppedFaces(frameBitmap) }
        BatchedFileLogger.log("Time Taken for Face Detection: ${t1.toLong(DurationUnit.MILLISECONDS)} MILLISECONDS")
        val faceRecognitionResults = ArrayList<FaceRecognitionResult>()
        var avgT2 = 0L
        var avgT3 = 0L
        var avgT4 = 0L

        for (result in faceDetectionResult) {
            // Get the embedding for the cropped face (query embedding)
            val (croppedBitmap, boundingBox,trackingId) = result
            val blurry = isFrameBlurry(croppedBitmap)
            if (blurry) {
                faceRecognitionResults.add(FaceRecognitionResult("Blurry", boundingBox))
                continue
            }
            val (embedding, t2) = measureTimedValue { faceNet.getFaceEmbedding(croppedBitmap) }
//            insightFaceEmbeddingsExtractor.isReady()
//            val (embedding, t2) = measureTimedValue { insightFaceEmbeddingsExtractor.extractEmbedding(croppedBitmap)}

//            insightFaceBuffaloLarge.initialize()
                // Extract embedding from cropped face
//            val embedding = insightFaceBuffaloLarge.extractEmbedding(croppedBitmap)
//            val (embedding, t2) = measureTimedValue { insightFaceBuffaloLarge.extractEmbedding(croppedBitmap) }
//            Log.d("ImageVectorUseCase", embedding.toString())
            // Compare with another embedding
//            val similarity = insightFaceBuffaloLarge.cosineSimilarity(embedding1, embedding2)

            // Clean up when done
//            insightFaceBuffaloLarge.cleanup()
//            val (embedding, t2) = measureTimedValue { insightFaceBuffaloLarge.extractEmbedding(croppedBitmap) }

            BatchedFileLogger.log("Time Taken for embedding: ${t2.toLong(DurationUnit.MILLISECONDS)} MILLISECONDS")
            avgT2 += t2.toLong(DurationUnit.MILLISECONDS)
            // Perform nearest-neighbor search
            val (recognitionResult, t3) =
                measureTimedValue { imagesVectorDB.getNearestEmbeddingPersonName(embedding!!) }
            val embeddingRef = recognitionResult?.faceEmbedding
            val l2normEmb = l2Norm(embedding!!)
            val l2normRef = l2Norm(embeddingRef!!)
            BatchedFileLogger.log("L2 Norm: $l2normEmb $l2normRef")
            BatchedFileLogger.log("Time Taken to Perform nearest-neighbor search: ${t3.toLong(DurationUnit.MILLISECONDS)} MILLISECONDS")
            avgT3 += t3.toLong(DurationUnit.MILLISECONDS)
            if (recognitionResult == null) {
                faceRecognitionResults.add(FaceRecognitionResult("Not recognized", boundingBox))
                BatchedFileLogger.log("Face Not Found in nearest-neighbor search")
                continue
            }

            // Calculate cosine similarity between the nearest-neighbor
            // and the query embedding
//            val (distance,tDistance) = measureTimedValue{ euclideanDistance(embedding, recognitionResult.faceEmbedding) }
//            val (distance,tDistance) = measureTimedValue{ cosineDistance(embedding, recognitionResult.faceEmbedding) }
            val (distance,tDistance) = measureTimedValue{ calculateSimilarity(embedding, recognitionResult.faceEmbedding) }

            // If the distance > 0.4, we recognize the person
            // else we conclude that the face does not match enough
            BatchedFileLogger.log("Euclidean Distance: $distance ${recognitionResult.personName} and with time ${tDistance.toLong(DurationUnit.MILLISECONDS)} MILLISECONDS")
            if (distance > thresholdRepo.getThreshold()) {
//                val (finalPersonId,tAggregator) = measureTimedValue { identityAggregatorRepository.faceDetected(trackingId,recognitionResult.personID) }
//                BatchedFileLogger.log("Time Taken for Aggregator: ${tAggregator.toLong(DurationUnit.MILLISECONDS)}")
//                if (finalPersonId==-1L){
//                    faceRecognitionResults.add(
//                        FaceRecognitionResult("Recognizing", boundingBox, null)
//                    )
//                    continue
//                }
                val spoofResult = faceSpoofDetector.detectSpoof(frameBitmap, boundingBox)
                BatchedFileLogger.log("Time Taken for spoof detection: ${spoofResult.timeMillis} MILLISECONDS")
                avgT4 += spoofResult.timeMillis
//                if (recognitionResult.personID != finalPersonId) {
//                    val person = personDB.getPerson(finalPersonId)
//                    if (person != null) {
//                        faceRecognitionResults.add(
//                            FaceRecognitionResult(person.personName, boundingBox, spoofResult)
//                        )
//                        BatchedFileLogger.log("Face Identified with name: ${person.personName} and Distance $distance")
//                        BatchedFileLogger.log("Spoof Result: ${spoofResult.isSpoof} and Spoof Score:  ${spoofResult.score} ${person.personName} ")
//
//                    }
//                } else {
                faceRecognitionResults.add(
                    FaceRecognitionResult(
                        recognitionResult.personName,
                        boundingBox,
                        spoofResult
                    )
                )
                BatchedFileLogger.log("Face Identified with name: ${recognitionResult.personName} and Distance $distance")
                BatchedFileLogger.log("Spoof Result: ${spoofResult.isSpoof} and Spoof Score:  ${spoofResult.score} ${recognitionResult.personName} ")
            }
            else
            {
//            } else {
                faceRecognitionResults.add(
                    FaceRecognitionResult("Not recognized", boundingBox, null)
                )
//                identityAggregatorRepository.clearFace(trackingId)

//                BatchedFileLogger.log("Spoof Result: ${spoofResult.isSpoof} and Spoof Score:  ${spoofResult.score} Not recognized")
            }
        }
        val metrics =
            if (faceDetectionResult.isNotEmpty()) {
                RecognitionMetrics(
                    timeFaceDetection = t1.toLong(DurationUnit.MILLISECONDS),
                    timeFaceEmbedding = avgT2 / faceDetectionResult.size,
                    timeVectorSearch = avgT3 / faceDetectionResult.size,
                    timeFaceSpoofDetection = avgT4 / faceDetectionResult.size
                )
            } else {
                null
            }

        return Pair(metrics, faceRecognitionResults)
    }

    private fun cosineDistance(x1: FloatArray, x2: FloatArray): Float {
        var mag1 = 0.0f
        var mag2 = 0.0f
        var product = 0.0f
        for (i in x1.indices) {
            mag1 += x1[i].pow(2)
            mag2 += x2[i].pow(2)
            product += x1[i] * x2[i]
        }
        mag1 = sqrt(mag1)
        mag2 = sqrt(mag2)
        return product / (mag1 * mag2)
    }

    private fun euclideanDistance(x1: FloatArray, x2: FloatArray): Float {
        var distance = 0f
        for (i in x1.indices) {
            val diff: Float = x1[i] - x2[i]
            distance += diff * diff
        }
        return sqrt(distance.toDouble()).toFloat()
    }

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

    fun removeImages(personID: Long) {
        imagesVectorDB.removeFaceRecordsWithPersonID(personID)
    }

//    private fun isFrameBlurry(bitmap: Bitmap): Boolean {
//        val BLUR_THRESHOLD = 100.0 // Adjust this value based on testing
//        val MIN_IMAGE_SIZE = 200 // Minimum size for blur detection
//        // Convert bitmap to Mat
////        val stream = ByteArrayOutputStream()
////        bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
////        val byteArray = stream.toByteArray()
////        val matOfByte = MatOfByte(*byteArray)
////        val mat = Imgcodecs.imdecode(matOfByte, Imgcodecs.IMREAD_GRAYSCALE)
//
//        val mat = Mat()
//        Utils.bitmapToMat(bitmap, mat)
//        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2GRAY)
//
//        // Resize if image is too large
//        if (mat.width() > MIN_IMAGE_SIZE || mat.height() > MIN_IMAGE_SIZE) {
//            val scale = MIN_IMAGE_SIZE.toDouble() / mat.width().coerceAtLeast(mat.height())
//            Imgproc.resize(
//                mat,
//                mat,
//                org.opencv.core.Size(
//                    mat.width() * scale,
//                    mat.height() * scale
//                )
//            )
//        }
//
//        // Calculate Laplacian variance
//        val destination = MatOfDouble()
//        Imgproc.Laplacian(mat, destination,  org.opencv.core.CvType.CV_64F)
//        val median = MatOfDouble()
//        Core.meanStdDev(destination, median, MatOfDouble())
//        val variance = Math.pow(median.get(0, 0)[0], 2.0)
//
//        // Clean up
//        mat.release()
//        destination.release()
//        median.release()
//        Log.d("ImageVectorUseCase", "Variance: $variance")
//
//        return variance < BLUR_THRESHOLD
//    }


    private fun isFrameBlurry(bitmap: Bitmap): Boolean {
        val BLUR_THRESHOLD = 400.0    // Tune this by experiment
        val MIN_IMAGE_SIZE = 200      // Downscale limit for speed

        // 1. Convert Bitmap → Mat (RGBA) → Gray
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2GRAY)

        // 2. Downscale if larger than MIN_IMAGE_SIZE (for speed)
        val maxDim = mat.width().coerceAtLeast(mat.height())
        if (maxDim > MIN_IMAGE_SIZE) {
            val scale = MIN_IMAGE_SIZE.toDouble() / maxDim
            val newW = (mat.width() * scale).toInt()
            val newH = (mat.height() * scale).toInt()
            Imgproc.resize(mat, mat, org.opencv.core.Size(newW.toDouble(), newH.toDouble()))
        }

        // 3. Compute Laplacian into a Mat (CV_64F for precision)
        val laplacian = Mat()
        Imgproc.Laplacian(mat, laplacian, org.opencv.core.CvType.CV_64F)

        // 4. Compute mean and stddev of the Laplacian response
        val mean = MatOfDouble()
        val stddev = MatOfDouble()
        Core.meanStdDev(laplacian, mean, stddev)

        // 5. Laplacian variance = (stddev)^2
        val sigma = stddev[0, 0][0]         // standard deviation
        val variance = sigma * sigma

        // 6. Release Mats
        mat.release()
        laplacian.release()
        mean.release()
        stddev.release()

        Log.d("FaceBlurCheck", "Laplacian variance = $variance")
        return variance < BLUR_THRESHOLD
    }

}
