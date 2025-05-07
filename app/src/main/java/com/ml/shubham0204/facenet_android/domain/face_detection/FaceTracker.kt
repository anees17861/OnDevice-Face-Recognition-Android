package com.ml.shubham0204.facenet_android.domain.face_detection
import android.graphics.Rect
import com.ml.shubham0204.facenet_android.util.BatchedFileLogger
import org.koin.core.annotation.Single
import kotlin.math.max
import kotlin.math.min

@Single
class FaceTracker {
    private var nextTrackId = 0
    private var trackedFaces = mutableMapOf<Int, TrackedFace>()
    private val maxTrackAge = 10 // Maximum number of frames a track can be lost before being removed
    private val iouThreshold = 0.5f // Minimum IoU for matching faces

    data class TrackedFace(
        val trackId: Int,
        var boundingBox: Rect,
        var lostFrames: Int = 0,
        var age: Int = 0
    )

    fun updateTracks(detectedFaces: List<Rect>): List<Pair<Int, Rect>> {
        val assignedTracks = mutableSetOf<Int>()
        val assignedDetections = mutableSetOf<Int>()
        val matches = mutableListOf<Pair<Int, Int>>() // (trackIdx, detectionIdx)

        // Calculate IoU between all tracked faces and new detections
        for ((trackId, trackedFace) in trackedFaces) {
            for ((detectionIdx, detection) in detectedFaces.withIndex()) {
                val iou = calculateIoU(trackedFace.boundingBox, detection)
                if (iou > iouThreshold) {
                    matches.add(Pair(trackId, detectionIdx))
                }
            }
        }

        // Sort matches by IoU score (highest first)
        matches.sortByDescending { (trackId, detectionIdx) ->
            calculateIoU(trackedFaces[trackId]!!.boundingBox, detectedFaces[detectionIdx])
        }

        // Update matched tracks
        for ((trackId, detectionIdx) in matches) {
            if (trackId !in assignedTracks && detectionIdx !in assignedDetections) {
                assignedTracks.add(trackId)
                assignedDetections.add(detectionIdx)

                trackedFaces[trackId]?.apply {
                    boundingBox = detectedFaces[detectionIdx]
                    lostFrames = 0
                    age++
                }
            }
        }

        // Create new tracks for unmatched detections
        for ((idx, detection) in detectedFaces.withIndex()) {
            if (idx !in assignedDetections) {
                BatchedFileLogger.log("New Face Added for tracking")
                val newTrackId = nextTrackId++
                trackedFaces[newTrackId] = TrackedFace(newTrackId, detection)
            }
        }

        // Update lost tracks and remove old ones
        val tracksToRemove = mutableListOf<Int>()
        val tracksToShadow = mutableListOf<Int>()

        for ((trackId, track) in trackedFaces) {
            if (trackId !in assignedTracks) {
                track.lostFrames++
                tracksToShadow.add(trackId)
                if (track.lostFrames > maxTrackAge) {
                    tracksToRemove.add(trackId)
                }
            }
        }
        tracksToRemove.forEach { trackedFaces.remove(it) }

        // Return current tracks
        return trackedFaces
            .filter { tracksToShadow.contains(it.key).not() }
            .map { (trackId, track) ->
            Pair(trackId, track.boundingBox)
        }
    }

    private fun calculateIoU(box1: Rect, box2: Rect): Float {
        val intersectionLeft = max(box1.left, box2.left)
        val intersectionTop = max(box1.top, box2.top)
        val intersectionRight = min(box1.right, box2.right)
        val intersectionBottom = min(box1.bottom, box2.bottom)

        if (intersectionLeft < intersectionRight && intersectionTop < intersectionBottom) {
            val intersection = (intersectionRight - intersectionLeft) *
                    (intersectionBottom - intersectionTop)
            val box1Area = box1.width() * box1.height()
            val box2Area = box2.width() * box2.height()
            val union = box1Area + box2Area - intersection
            return intersection.toFloat() / union.toFloat()
        }
        return 0f
    }

    fun reset() {
        trackedFaces.clear()
        nextTrackId = 0
    }
}