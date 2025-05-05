package com.ml.shubham0204.facenet_android.domain.face_detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Rect
import android.net.Uri
import androidx.compose.ui.graphics.toComposeRect
import androidx.core.graphics.toRect
import androidx.core.graphics.toRectF
import androidx.exifinterface.media.ExifInterface
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.ml.shubham0204.facenet_android.domain.AppException
import com.ml.shubham0204.facenet_android.domain.ErrorCode
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import org.koin.core.annotation.Single
import java.io.File
import java.io.FileOutputStream

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
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .build()
    private val faceDetector = FaceDetection.getClient(highAccuracyOpts)

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
    suspend fun getAllCroppedFaces(frameBitmap: Bitmap): List<Pair<Bitmap, Rect>> =
        withContext(Dispatchers.IO) {
            return@withContext detectProcess(faceDetector, InputImage.fromBitmap(frameBitmap, 0))
                .filter { validateRect(frameBitmap, it.boundingBox) }
                .map { detection -> detection.boundingBox }
                .map { rect ->
                    val croppedBitmap =
                        Bitmap.createBitmap(
                            frameBitmap,
                            rect.left,
                            rect.top,
                            rect.width(),
                            rect.height()
                        )
                    Pair(croppedBitmap, rect)
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

    private suspend fun detectProcess(faceDetector: FaceDetector, image: InputImage): List<Face> =
        suspendCancellableCoroutine {
            faceDetector.process(image)
                .addOnSuccessListener { faces -> it.resumeWith(Result.success(faces)) }
                .addOnFailureListener { e -> it.resumeWith(Result.failure(e)) }
        }


}
