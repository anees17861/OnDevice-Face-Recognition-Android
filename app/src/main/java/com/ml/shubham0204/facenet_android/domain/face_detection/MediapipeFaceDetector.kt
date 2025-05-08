package com.ml.shubham0204.facenet_android.domain.face_detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.RectF
import android.net.Uri
import android.util.Log
import androidx.core.graphics.toRect
import androidx.exifinterface.media.ExifInterface
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.Detection
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facedetector.FaceDetector
import com.ml.shubham0204.facenet_android.domain.AppException
import com.ml.shubham0204.facenet_android.domain.ErrorCode
import com.ml.shubham0204.facenet_android.util.BatchedFileLogger
import java.io.File
import java.io.FileOutputStream
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.koin.core.annotation.Single
import kotlin.math.atan2

// Utility class for interacting with Mediapipe's Face Detector
// See https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/android
@Single
class MediapipeFaceDetector(private val context: Context) {

    // The model is stored in the assets folder
    private val modelName = "blaze_face_short_range.tflite"
    private val confidenceThreshold = 0.9f
    private val minimumSuppressionThreshold = 0.5f
    private val baseOptions = BaseOptions.builder().setModelAssetPath(modelName).build()
    private val faceDetectorOptions =
        FaceDetector.FaceDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.IMAGE)
            .setMinDetectionConfidence(confidenceThreshold)
            .setMinSuppressionThreshold(minimumSuppressionThreshold)
            .build()
    private val faceDetector = FaceDetector.createFromOptions(context, faceDetectorOptions)
    private val faceTracker = FaceTracker()

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
//            saveBitmap(context,imageBitmap,"input_img")
            val faces = faceDetector.detect(BitmapImageBuilder(imageBitmap).build()).detections()
            if (faces.size > 1) {
                return@withContext Result.failure<Bitmap>(AppException(ErrorCode.MULTIPLE_FACES))
            } else if (faces.size == 0) {
                return@withContext Result.failure<Bitmap>(AppException(ErrorCode.NO_FACE))
            } else {
                // Validate the bounding box and
                // return the cropped face
                val rect = faces[0].boundingBox().toRect()
                if (validateRect(imageBitmap, rect)) {
                    val croppedBitmap =
                        Bitmap.createBitmap(
                            imageBitmap,
                            rect.left,
                            rect.top,
                            rect.width(),
                            rect.height()
                        )
                    val alignedFace = alignFace(imageBitmap, faces[0])
//                    saveBitmap(context,alignedFace,"test_input" )
                    return@withContext Result.success(croppedBitmap)
                } else {
                    return@withContext Result.failure<Bitmap>(
                        AppException(ErrorCode.FACE_DETECTOR_FAILURE)
                    )
                }
            }
        }

    // Detects multiple faces from the `frameBitmap`
    // and returns pairs of (croppedFace , boundingBoxRect)
    // Used by ImageVectorUseCase.kt
//    suspend fun getAllCroppedFaces(frameBitmap: Bitmap): List<Pair<Bitmap, Rect>> =
//        withContext(Dispatchers.IO) {
//
//            return@withContext faceDetector
//                .detect(BitmapImageBuilder(frameBitmap).build())
//                .detections()
//                .filter { validateRect(frameBitmap, it.boundingBox().toRect()) }
//                .map {
//                    detection ->
//                    detection.boundingBox().toRect() }
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

    suspend fun getAllCroppedFaces(frameBitmap: Bitmap): List<Triple<Bitmap, Rect,Int>> =
        withContext(Dispatchers.IO) {
            // Get face detections
            val detectedFaces = faceDetector
                .detect(BitmapImageBuilder(frameBitmap).build())
                .detections()
//                .map { it.boundingBox().toRect() }
                .filter { validateRect(frameBitmap, it.boundingBox().toRect()) }

            // Update tracks
            val trackedFaces = faceTracker.updateTracks(detectedFaces)


//            BatchedFileLogger.log("Tracked Faces and Bounding Boxes in a frame: $trackedFaces")
            // Return cropped faces with their tracking IDs
            return@withContext trackedFaces.map { (id, detection) ->
                val rect = detection.boundingBox().toRect()
                val alignedFace = alignFace(frameBitmap, detection)
//                saveBitmap(context,alignedFace, id.toString())
                val croppedBitmap = Bitmap.createBitmap(
                    frameBitmap,
                    rect.left,
                    rect.top,
                    rect.width(),
                    rect.height()
                )

                Triple(croppedBitmap, rect,id)
            }
        }

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

    private fun alignFace(
        bitmap: Bitmap,
        detection: Detection
    ): Bitmap {
        // 1) Get normalized eye keypoints
        if (!detection.keypoints().isPresent and (detection.keypoints().get().size < 2)) {
            return bitmap
        }
        val leftEye = detection.keypoints().get()[0]
        val rightEye = detection.keypoints().get()[1]
        if (rightEye == null || leftEye == null) return bitmap

        // 2. Calculate rotation angle from normalized coordinates
        val dx = (rightEye.x() - leftEye.x())* bitmap.width
        val dy = (rightEye.y() - leftEye.y())* bitmap.height
        Log.d("dxdy", "dx: $dx, dy: $dy $rightEye $leftEye")
        val angle = Math.toDegrees(atan2(dy.toDouble(), dx.toDouble())).toFloat()

        // 3. Get face bounding box with margin
        val boundingBox = detection.boundingBox()

        // 4. Convert normalized coordinates to pixel values
//        val margin = 0.1f // 10% margin
        val margin = minOf(boundingBox.width(), boundingBox.height()) * 0.2f
        val pixelLeft = ((boundingBox.left - margin)).toInt().coerceAtLeast(0)
        val pixelTop = ((boundingBox.top - margin)).toInt().coerceAtLeast(0)
        val pixelWidth = (boundingBox.width() + margin).toInt().coerceAtMost(bitmap.width)
        val pixelHeight = (boundingBox.height() + margin).toInt().coerceAtMost(bitmap.height)
        Log.d("PixelValues", "Left: $pixelLeft, Top: $pixelTop, Width: $pixelWidth, Height: $pixelHeight ")
        // 5. Create tight crop around face

        val faceCrop =
//            try {
            Bitmap.createBitmap(
                bitmap,
                pixelLeft,
                pixelTop,
                pixelWidth,
                pixelHeight
            )
//        } catch (e: IllegalArgumentException) {
//            return bitmap
//        }

        // 6. Calculate rotation pivot point (eye midpoint relative to crop)
        val eyeMidpointX = ((leftEye.x() + rightEye.x()) / 2f) * bitmap.width
        val eyeMidpointY = ((leftEye.y() + rightEye.y()) / 2f) * bitmap.height
        Log.d("eyeMidpointValues", "X: $eyeMidpointX, Y: $eyeMidpointY")
        Log.d("Angle", "Angle: $angle")
//        Log.d("bitmap", "Width: ${bitmap.width}, Height: ${bitmap.height}")

        // 7. Create rotation matrix around eye midpoint
        val matrix = Matrix().apply {
            postRotate(-angle, faceCrop.width / 2f, faceCrop.height / 2f)
        }
        val rotatedBitmap =
            Bitmap.createBitmap(faceCrop, 0, 0, faceCrop.width, faceCrop.height, matrix, true)

        return rotatedBitmap


    }
}
