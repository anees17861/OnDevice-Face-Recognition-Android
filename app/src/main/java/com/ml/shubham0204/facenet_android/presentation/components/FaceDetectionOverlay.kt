package com.ml.shubham0204.facenet_android.presentation.components

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.widget.FrameLayout
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.graphics.createBitmap
import androidx.core.graphics.toRectF
import androidx.core.view.doOnLayout
import androidx.lifecycle.LifecycleOwner
import com.ml.shubham0204.facenet_android.presentation.screens.detect_screen.DetectScreenViewModel
import com.ml.shubham0204.facenet_android.utils.ImageUtils
import java.util.concurrent.Executors
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.imgproc.Imgproc

@SuppressLint("ViewConstructor")
@ExperimentalGetImage
class FaceDetectionOverlay(
    private val lifecycleOwner: LifecycleOwner,
    private val context: Context,
    private val viewModel: DetectScreenViewModel
) : FrameLayout(context) {

    private var overlayWidth: Int = 0
    private var overlayHeight: Int = 0

    private var imageTransform: Matrix = Matrix()
    private var boundingBoxTransform: Matrix = Matrix()
    private var isImageTransformedInitialized = false
    private var isBoundingBoxTransformedInitialized = false

    private lateinit var sourceBitmap: Bitmap
    private lateinit var frameBitmap: Bitmap
    private var isProcessing = false
    private var cameraFacing: Int = CameraSelector.LENS_FACING_BACK
    private lateinit var boundingBoxOverlay: BoundingBoxOverlay
    private lateinit var previewView: PreviewView

    var predictions: Array<Prediction> = arrayOf()

    init {
        initializeCamera(cameraFacing)
        doOnLayout {
            overlayHeight = it.measuredHeight
            overlayWidth = it.measuredWidth
        }
    }

    fun initializeCamera(cameraFacing: Int) {
        this.cameraFacing = cameraFacing
        this.isImageTransformedInitialized = false
        this.isBoundingBoxTransformedInitialized = false
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        val previewView = PreviewView(context)
        val executor = ContextCompat.getMainExecutor(context)
        cameraProviderFuture.addListener(
            {
                val cameraProvider = cameraProviderFuture.get()
                val preview =
                    Preview.Builder().build().also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }
                val cameraSelector =
                    CameraSelector.Builder().requireLensFacing(cameraFacing).build()
                val frameAnalyzer =
                    ImageAnalysis.Builder()
                        .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
//                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .build()
                frameAnalyzer.setAnalyzer(Executors.newSingleThreadExecutor(), analyzer)
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    preview,
                    frameAnalyzer
                )
            },
            executor
        )
        if (childCount == 2) {
            removeView(this.previewView)
            removeView(this.boundingBoxOverlay)
        }
        this.previewView = previewView
        addView(this.previewView)

        val boundingBoxOverlayParams =
            LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT)
        this.boundingBoxOverlay = BoundingBoxOverlay(context)
        this.boundingBoxOverlay.setWillNotDraw(false)
        this.boundingBoxOverlay.setZOrderOnTop(true)
        addView(this.boundingBoxOverlay, boundingBoxOverlayParams)
    }

    private val BLUR_THRESHOLD = 100.0 // Adjust this value based on testing
    private val MIN_IMAGE_SIZE = 200 // Minimum size for blur detection

    private fun isFrameBlurry(bitmap: Bitmap): Boolean {
        // Convert bitmap to Mat
//        val stream = ByteArrayOutputStream()
//        bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
//        val byteArray = stream.toByteArray()
//        val matOfByte = MatOfByte(*byteArray)
//        val mat = Imgcodecs.imdecode(matOfByte, Imgcodecs.IMREAD_GRAYSCALE)

        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2GRAY)

        // Resize if image is too large
        if (mat.width() > MIN_IMAGE_SIZE || mat.height() > MIN_IMAGE_SIZE) {
            val scale = MIN_IMAGE_SIZE.toDouble() / mat.width().coerceAtLeast(mat.height())
            Imgproc.resize(
                mat,
                mat,
                org.opencv.core.Size(
                    mat.width() * scale,
                    mat.height() * scale
                )
            )
        }

        // Calculate Laplacian variance
        val destination = MatOfDouble()
        Imgproc.Laplacian(mat, destination,  org.opencv.core.CvType.CV_64F)
        val median = MatOfDouble()
        Core.meanStdDev(destination, median, MatOfDouble())
        val variance = Math.pow(median.get(0, 0)[0], 2.0)

        // Clean up
        mat.release()
        destination.release()
        median.release()

        return variance < BLUR_THRESHOLD
    }

    private val analyzer =
        ImageAnalysis.Analyzer { image ->
            if (isProcessing) {
                image.close()
                return@Analyzer
            }
            isProcessing = true

            // Transform android.net.Image to Bitmap
            sourceBitmap =
                Bitmap.createBitmap(
                    image.image!!.width,
                    image.image!!.height,
                    Bitmap.Config.ARGB_8888
                )
            if (image.format == ImageFormat.YUV_420_888) {

                var rgbArray = IntArray(image.width * image.height)
                ImageUtils.convertYUVToARGB(image, rgbArray)
                sourceBitmap.setPixels(
                    rgbArray,
                    0,
                    sourceBitmap.width,
                    0,
                    0,
                    sourceBitmap.width,
                    sourceBitmap.height
                )
            } else {
                sourceBitmap.copyPixelsFromBuffer(image.planes[0].buffer)
            }

            // Configure frameHeight and frameWidth for output2overlay transformation matrix
            // and apply it to `frameBitmap`
            if (!isImageTransformedInitialized) {
                imageTransform = Matrix()
                imageTransform.apply { postRotate(image.imageInfo.rotationDegrees.toFloat()) }
                isImageTransformedInitialized = true
            }
            frameBitmap =
                Bitmap.createBitmap(
                    sourceBitmap,
                    0,
                    0,
                    sourceBitmap.width,
                    sourceBitmap.height,
                    imageTransform,
                    false
                )

            if (!isBoundingBoxTransformedInitialized) {
                boundingBoxTransform = Matrix()
                boundingBoxTransform.apply {
                    setScale(
                        overlayWidth / frameBitmap.width.toFloat(),
                        overlayHeight / frameBitmap.height.toFloat()
                    )
                    if (cameraFacing == CameraSelector.LENS_FACING_FRONT) {
                        // Mirror the bounding box coordinates
                        // for front-facing camera
                        postScale(
                            -1f,
                            1f,
                            overlayWidth.toFloat() / 2.0f,
                            overlayHeight.toFloat() / 2.0f
                        )
                    }
                }
                isBoundingBoxTransformedInitialized = true
            }
            CoroutineScope(Dispatchers.Default).launch {
                val predictions = ArrayList<Prediction>()
                val (metrics, results) = viewModel.imageVectorUseCase.getNearestPersonName(frameBitmap)
                results.forEach {
                    (name, boundingBox, spoofResult) ->
                    val box = boundingBox.toRectF()
                    var personName = name
                    if (viewModel.getNumPeople().toInt() == 0) {
                        personName = ""
                    }
                    if (spoofResult != null && (spoofResult.isSpoof || spoofResult.score<0.80)) {
//                        personName = "$personName (Spoof: ${spoofResult.score})"
                          personName = "Spoof"
                    }else if (personName.isNotEmpty() && personName != "Not recognized" && personName != "Recognizing") {
//                        personName == "Not recognized" || personName == "Recognizing"
                        // Trigger recognition event when a face is recognized
                        viewModel.showRecognition(name)
                    }
                    boundingBoxTransform.mapRect(box)
                    predictions.add(Prediction(box, personName))
                }
                withContext(Dispatchers.Main) {
                    viewModel.faceDetectionMetricsState.value = metrics
                    this@FaceDetectionOverlay.predictions = predictions.toTypedArray()
                    boundingBoxOverlay.invalidate()
                    isProcessing = false
                }
            }
            image.close()
        }

    data class Prediction(var bbox: RectF, var label: String)

    inner class BoundingBoxOverlay(context: Context) :
        SurfaceView(context), SurfaceHolder.Callback {

//        private val boxPaint =
//            Paint().apply {
//                color = Color.parseColor("#4D90caf9")
//                style = Paint.Style.FILL
//            }
//        private val textPaint =
//            Paint().apply {
//                strokeWidth = 2.0f
//                textSize = 36f
//                color = Color.WHITE
//            }

        private val normalBoxPaint =
            Paint().apply {
                color = Color.parseColor("#4D90caf9")
                style = Paint.Style.FILL
            }

        private val spoofBoxPaint =
            Paint().apply {
                color = Color.parseColor("#4DFF0000") // Semi-transparent red
                style = Paint.Style.FILL
            }

        private val recognizedBoxPaint =
            Paint().apply {
                color = Color.parseColor("#4D00FF00") // Semi-transparent green
                style = Paint.Style.FILL
            }

        private val textPaint =
            Paint().apply {
                strokeWidth = 2.0f
                textSize = 36f
                color = Color.WHITE
            }

        override fun surfaceCreated(holder: SurfaceHolder) {}

        override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {}

        override fun surfaceDestroyed(holder: SurfaceHolder) {}

        override fun onDraw(canvas: Canvas) {
            predictions.forEach {
//                canvas.drawRoundRect(it.bbox, 16f, 16f, boxPaint)
//                canvas.drawText(it.label, it.bbox.centerX(), it.bbox.centerY(), textPaint)
                val isSpoof = it.label.contains("Spoof")
                val isRecognizing = it.label.contains("Recognizing")
                val isNoTRecognized = it.label.contains("Not recognized")
//                val paint = if (isSpoof) spoofBoxPaint else normalBoxPaint
                val paint = when {
                    isSpoof-> spoofBoxPaint
                    it.label == "Not recognized" || it.label == "Recognizing" -> normalBoxPaint
                    else -> recognizedBoxPaint // Has a name = recognized face
                }

                canvas.drawRoundRect(it.bbox, 16f, 16f, paint)
                canvas.drawText(it.label, it.bbox.centerX(), it.bbox.centerY(), textPaint)

            }
        }
    }
}
