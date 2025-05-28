package com.ml.shubham0204.facenet_android.domain.face_detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.PointF
import android.graphics.Rect
import android.net.Uri
import android.util.Log
import androidx.compose.ui.graphics.toComposeRect
import androidx.core.graphics.scale
import androidx.core.graphics.toRect
import androidx.core.graphics.toRectF
import androidx.exifinterface.media.ExifInterface
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import com.ml.shubham0204.facenet_android.domain.AppException
import com.ml.shubham0204.facenet_android.domain.ErrorCode
import com.ml.shubham0204.facenet_android.util.BatchedFileLogger
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import org.koin.core.annotation.Single
import org.koin.core.time.measureTimedValue
import java.io.File
import java.io.FileOutputStream

import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.android.Utils

import org.tensorflow.lite.support.image.ops.ResizeOp
import kotlin.math.atan2
import kotlin.time.DurationUnit

// Utility class for interacting with MLKit's Face Detector
// See https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/android
@Single
class MLKitFaceDetector(private val context: Context) {

    // The model is stored in the assets folder
    //TODO initialize face detector
    // Multiple object detection in static images
    var highAccuracyOpts: FaceDetectorOptions =
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
//            .enableTracking()
            .build()
    private val faceDetector = FaceDetection.getClient(highAccuracyOpts)
//    private val faceTracker = FaceTracker()


    suspend fun getCroppedFace(imageUri: Uri): Result<Bitmap> =
        withContext(Dispatchers.IO) {
            var imageInputStream =
                context.contentResolver.openInputStream(imageUri)
                    ?: return@withContext Result.failure<Bitmap>(
                        AppException(ErrorCode.FACE_DETECTOR_FAILURE)
                    )
            var imageBitmap = BitmapFactory.decodeStream(imageInputStream)
            imageInputStream.close()

            // Re-create an input-stream to reset its position
            // InputStream returns false with markSupported(), hence we cannot
            // reset its position
            // Without recreating the inputStream, no exif-data is read
            imageInputStream =
                context.contentResolver.openInputStream(imageUri)
                    ?: return@withContext Result.failure<Bitmap>(
                        AppException(ErrorCode.FACE_DETECTOR_FAILURE)
                    )
            val exifInterface = ExifInterface(imageInputStream)
            imageBitmap =
                when (
                    exifInterface.getAttributeInt(
                        ExifInterface.TAG_ORIENTATION,
                        ExifInterface.ORIENTATION_UNDEFINED
                    )
                ) {
                    ExifInterface.ORIENTATION_ROTATE_90 -> rotateBitmap(imageBitmap, 90f)
                    ExifInterface.ORIENTATION_ROTATE_180 -> rotateBitmap(imageBitmap, 180f)
                    ExifInterface.ORIENTATION_ROTATE_270 -> rotateBitmap(imageBitmap, 270f)
                    else -> imageBitmap
                }
            imageInputStream.close()

            // We need exactly one face in the image, in other cases, return the
            // necessary errors
            val image = InputImage.fromBitmap(imageBitmap, 0)
            val faces = detectProcess(faceDetector, image)
//            val faces = faceDetector.detect(BitmapImageBuilder(imageBitmap).build()).detections()
            if (faces.size > 1) {
                return@withContext Result.failure<Bitmap>(AppException(ErrorCode.MULTIPLE_FACES))
            } else if (faces.isEmpty()) {
                return@withContext Result.failure<Bitmap>(AppException(ErrorCode.NO_FACE))
            } else {
                // Validate the bounding box and
                // return the cropped face
                val rect = faces[0].boundingBox
                if (validateRect(imageBitmap, rect)) {
                    val croppedBitmap =
                        Bitmap.createBitmap(
                            imageBitmap,
                            rect.left,
                            rect.top,
                            rect.width(),
                            rect.height()
                        )
//                    val aligned_face = alignFace(imageBitmap,faces[0],160)
                    val alignedFace = alignFaceUsing5Points(imageBitmap, faces[0])
                    return@withContext Result.success(alignedFace)
                } else {
                    return@withContext Result.failure<Bitmap>(
                        AppException(ErrorCode.FACE_DETECTOR_FAILURE)
                    )
                }
            }
        }



    private fun alignFace(bitmap: Bitmap, face: Face, targetSize: Int): Bitmap {
        // Get face landmarks if available
//        val leftEye = face.getLandmark(FaceLandmark.LEFT_EAR)?.position
//        val rightEye = face.getLandmark(FaceLandmark.RIGHT_EYE)?.position

        // Calculate rotation angle
//        val rotationDegrees = if (leftEye != null && rightEye != null) {
//            val deltaY = rightEye.y - leftEye.y
//            val deltaX = rightEye.x - leftEye.x
//            val radians = atan2(deltaY, deltaX)
//            Math.toDegrees(radians.toDouble()).toFloat()
//        } else {
//            // Use face rotation if landmarks not available
//            face.headEulerAngleZ
//        }
        val rotationDegrees = face.headEulerAngleZ
        // Calculate the center of the face
        val centerX = face.boundingBox.exactCenterX()
        val centerY = face.boundingBox.exactCenterY()

        // Calculate how much to pad around the face
        val width = face.boundingBox.width()
        val height = face.boundingBox.height()
        val paddingFactor = 0.4f // 40% padding
        val paddedWidth = width * (1 + paddingFactor)
        val paddedHeight = height * (1 + paddingFactor)

        // Create transformation matrix
        val matrix = Matrix()
        matrix.postRotate(-rotationDegrees, centerX, centerY)

        // Calculate the area to crop
        val cropStartX = (centerX - paddedWidth / 2).coerceAtLeast(0f).toInt()
        val cropStartY = (centerY - paddedHeight / 2).coerceAtLeast(0f).toInt()
        val cropWidth = paddedWidth.coerceAtMost(bitmap.width - cropStartX.toFloat()).toInt()
        val cropHeight = paddedHeight.coerceAtMost(bitmap.height - cropStartY.toFloat()).toInt()

        // Apply rotation
        val rotatedBitmap = Bitmap.createBitmap(
            bitmap,
            0, 0,
            bitmap.width, bitmap.height,
            matrix,
            true
        )

        // Crop the aligned face
        var croppedBitmap = Bitmap.createBitmap(
            rotatedBitmap,
            cropStartX, cropStartY,
            cropWidth, cropHeight
        )

        // Scale to target size
//        croppedBitmap = croppedBitmap.scale(targetSize, targetSize)

        // Clean up temporary bitmaps
        if (rotatedBitmap != bitmap && rotatedBitmap != croppedBitmap) {
            rotatedBitmap.recycle()
        }

        return croppedBitmap
    }

    // Detects multiple faces from the `frameBitmap`
    // and returns pairs of (croppedFace , boundingBoxRect)
    // Used by ImageVectorUseCase.kt
//    suspend fun getAllCroppedFaces(frameBitmap: Bitmap): List<Triple<Bitmap, Rect,Int>> =
//        withContext(Dispatchers.IO) {
////            val scaledbitmap = Bitmap.createScaledBitmap(frameBitmap,480, 360, true)
//            return@withContext detectProcess(faceDetector, InputImage.fromBitmap(frameBitmap, 0))
//                .filter { validateRect(frameBitmap, it.boundingBox) }
//                .map { detection ->
////                    if (detection.trackingId != null) {
//                    val id = 1
////                    val id = detection.trackingId
////                    val rotY = detection.headEulerAngleY // Head is rotated to the right rotY degrees
////                    val rotZ = detection.headEulerAngleZ // Head is tilted sideways rotZ degrees
////                    val rotX = detection.headEulerAngleX // Head is rotated to the right rotY degrees
////                        val rotZ = detection.headEulerAngleZ // Head is tilted sideways rotZ degrees
////                    val aligned_face = alignFace(frameBitmap,detection,160)
//                    val alignedFace = alignFaceUsing5Points(frameBitmap, detection)
//                    Triple(alignedFace, detection.boundingBox,id!!)
//                }
//
//        }
    suspend fun getAllCroppedFaces(frameBitmap: Bitmap): List<Triple<Bitmap, Rect, Int>> =
        withContext(Dispatchers.IO) {
            val faces = detectProcess(faceDetector, InputImage.fromBitmap(frameBitmap, 0))
                .filter { validateRect(frameBitmap, it.boundingBox) }

            // Find the largest face by comparing bounding box areas
            val ( largestFace,tLargestFace ) = measureTimedValue {   faces.maxByOrNull { face ->
                face.boundingBox.width() * face.boundingBox.height()
            } }
            Log.d("MLKitFaceDetector", "Time Taken for Largest Face: ${tLargestFace} MILLISECONDS")


            return@withContext if (largestFace != null) {
                val (alignedFace,tAlignedFace) = measureTimedValue{alignFaceUsing5Points(frameBitmap, largestFace)}
                Log.d("MLKitFaceDetector", "Time Taken for Aligned Face: ${tAlignedFace} MILLISECONDS")
                listOf(Triple(alignedFace, largestFace.boundingBox, 1))
            } else {
                emptyList()
            }
        }
//                .map { rect ->
//                    val croppedBitmap =
//                        Bitmap.createBitmap(
//                            frameBitmap,
//                            rect.left,
//                            rect.top,
//                            rect.width(),
//                            rect.height()
//                        )
//                    Pair(croppedBitmap, rect)
//                }
//        }

    // DEBUG: For testing purpose, saves the Bitmap to the app's private storage
    fun saveBitmap(context: Context, image: Bitmap, name: String) {
        val fileOutputStream = FileOutputStream(File(context.filesDir.absolutePath + "/$name.png"))
        image.compress(Bitmap.CompressFormat.PNG, 100, fileOutputStream)
    }

    private fun rotateBitmap(source: Bitmap, degrees: Float): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(degrees)
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, false)
    }

    // Check if the bounds of `boundingBox` fit within the
    // limits of `cameraFrameBitmap`
    private fun validateRect(cameraFrameBitmap: Bitmap, boundingBox: Rect): Boolean {
        return boundingBox.left >= 0 &&
                boundingBox.top >= 0 &&
                (boundingBox.left + boundingBox.width()) < cameraFrameBitmap.width &&
                (boundingBox.top + boundingBox.height()) < cameraFrameBitmap.height
    }

    private suspend fun detectProcess(faceDetector: FaceDetector, image: InputImage): List<Face> =
        suspendCancellableCoroutine {
            faceDetector.process(image)
                .addOnSuccessListener { faces -> it.resumeWith(Result.success(faces)) }
                .addOnFailureListener { e -> it.resumeWith(Result.failure(e)) }
        }

    private fun alignFaceUsing5Points(bitmap: Bitmap, face: Face, targetSize: Int = 112): Bitmap {
        // Get the required face landmarks
        val leftEye = face.getLandmark(FaceLandmark.LEFT_EYE)?.position
        val rightEye = face.getLandmark(FaceLandmark.RIGHT_EYE)?.position
        val nose = face.getLandmark(FaceLandmark.NOSE_BASE)?.position
        val leftMouth = face.getLandmark(FaceLandmark.MOUTH_LEFT)?.position
        val rightMouth = face.getLandmark(FaceLandmark.MOUTH_RIGHT)?.position

        // Check if all landmarks are detected
        if (leftEye == null || rightEye == null || nose == null ||
            leftMouth == null || rightMouth == null) {
            return bitmap // Return original bitmap if landmarks are missing
        }

        try {
            // Source points (detected landmarks)
            val srcPoints = MatOfPoint2f()
            val srcPointsArray = arrayOf(
                Point(leftEye.x.toDouble(), leftEye.y.toDouble()),
                Point(rightEye.x.toDouble(), rightEye.y.toDouble()),
                Point(nose.x.toDouble(), nose.y.toDouble())
            )
            srcPoints.fromArray(*srcPointsArray)

            // Standard InsightFace/Buffalo-L reference points (normalized coordinates)
            val dstPoints = MatOfPoint2f()
            val dstPointsArray = arrayOf(
                Point(38.2946, 51.6963),  // Left eye
                Point(73.5318, 51.5014),  // Right eye
                Point(56.0252, 71.7366)   // Nose
            )
            dstPoints.fromArray(*dstPointsArray)

            // Convert bitmap to OpenCV Mat
            val sourceMat = Mat()
            Utils.bitmapToMat(bitmap, sourceMat)

            // Calculate transformation matrix using three points (eyes and nose)
            val transformMatrix = Imgproc.getAffineTransform(srcPoints, dstPoints)

            // Create output matrix and apply transformation
            val outputMat = Mat()
            Imgproc.warpAffine(
                sourceMat,
                outputMat,
                transformMatrix,
                Size(targetSize.toDouble(), targetSize.toDouble()),
                Imgproc.INTER_LINEAR
            )

            // Convert back to Bitmap
            val alignedBitmap = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(outputMat, alignedBitmap)

            // Cleanup
            sourceMat.release()
            outputMat.release()
            transformMatrix.release()
            srcPoints.release()
            dstPoints.release()

            return alignedBitmap
        } catch (e: Exception) {
            Log.e("MLKitFaceDetector", "Face alignment failed: ${e.message}")
            return bitmap
        }
    }
//    private fun alignFaceUsing5Points(bitmap: Bitmap, face: Face, targetSize: Int = 112): Bitmap {
//        // Get the required face landmarks
//        val leftEye = face.getLandmark(FaceLandmark.LEFT_EYE)?.position
//        val rightEye = face.getLandmark(FaceLandmark.RIGHT_EYE)?.position
//        val nose = face.getLandmark(FaceLandmark.NOSE_BASE)?.position
//        val leftMouth = face.getLandmark(FaceLandmark.MOUTH_LEFT)?.position
//        val rightMouth = face.getLandmark(FaceLandmark.MOUTH_RIGHT)?.position
//
//        // Check if all landmarks are detected
//        if (leftEye == null || rightEye == null || nose == null ||
//            leftMouth == null || rightMouth == null) {
//            return bitmap // Return original bitmap if landmarks are missing
//        }
//
//        // Source points (detected landmarks)
//        val srcPoints = MatOfPoint2f(
//            Point(leftEye.x.toDouble(), leftEye.y.toDouble()),
//            Point(rightEye.x.toDouble(), rightEye.y.toDouble()),
//            Point(nose.x.toDouble(), nose.y.toDouble()),
//            Point(leftMouth.x.toDouble(), leftMouth.y.toDouble()),
//            Point(rightMouth.x.toDouble(), rightMouth.y.toDouble())
//        )
//
//        // Standard InsightFace/Buffalo-L reference points (normalized coordinates)
//        val dstPoints = MatOfPoint2f(
//            Point(38.2946, 51.6963),  // Left eye
//            Point(73.5318, 51.5014),  // Right eye
//            Point(56.0252, 71.7366),  // Nose
//            Point(41.5493, 92.3655),  // Left mouth
//            Point(70.7299, 92.2041)   // Right mouth
//        )
//
//        try {
//            // Convert bitmap to OpenCV Mat
//            val sourceMat = Mat()
//            Utils.bitmapToMat(bitmap, sourceMat)
//
//            // Calculate transformation matrix
//            val transformMatrix = Imgproc.estimateAffine2D(srcPoints, dstPoints)
//            if (transformMatrix.empty()) {
//                return bitmap
//            }
//
//            // Create output matrix and apply transformation
//            val outputMat = Mat()
//            Imgproc.warpAffine(
//                sourceMat,
//                outputMat,
//                transformMatrix,
//                Size(targetSize.toDouble(), targetSize.toDouble()),
//                Imgproc.INTER_LINEAR
//            )
//
//            // Convert back to Bitmap
//            val alignedBitmap = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
//            Utils.matToBitmap(outputMat, alignedBitmap)
//
//            // Cleanup
//            sourceMat.release()
//            outputMat.release()
//            transformMatrix.release()
//            srcPoints.release()
//            dstPoints.release()
//
//            return alignedBitmap
//        } catch (e: Exception) {
//            Log.e("MLKitFaceDetector", "Face alignment failed: ${e.message}")
//            return bitmap
//        }
//    }


}
