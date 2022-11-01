package jp.co.smartbank.rectangledetector.strategy

import android.graphics.Bitmap
import android.util.Base64
import java.io.ByteArrayOutputStream
import org.opencv.android.Utils
import org.opencv.core.CvException
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * An implementation of [ContourDetectionStrategy] thresholding images.
 */
internal class AdaptiveThresholdStrategy : ContourDetectionStrategy() {
    override fun detectContours(originalImageMat: Mat): List<MatOfPoint> {
        val grayScaleMat = convertToGrayScale(originalImageMat)
        val thresholdingMat = thresholdImage(grayScaleMat)
        val noiseReducedMat = reduceNoises(thresholdingMat)
        val result = Mat()
        Imgproc.morphologyEx( //todo: extract
            noiseReducedMat, result, Imgproc.MORPH_CLOSE,
            Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(15.0, 15.0))
        )
        convertMatToBitMap(result) //todo: remove
        return findContours(result, allowanceRatioToArcLength = 0.02)
    }


    private fun convertToGrayScale(mat: Mat): Mat {
        val result = Mat()
        Imgproc.cvtColor(mat, result, Imgproc.COLOR_BGR2GRAY)
        return result
    }

    private fun thresholdImage(mat: Mat): Mat {
        val result = Mat()
        Imgproc.adaptiveThreshold(
            mat, result, 255.0, Imgproc.ADAPTIVE_THRESH_MEAN_C,
            Imgproc.THRESH_BINARY_INV, 91, 3.0
        )
        return result
    }

    private fun reduceNoises(mat: Mat): Mat {
        val result = Mat()
        Imgproc.medianBlur(mat, result, 7)
        return result
    }

    private fun convertMatToBitMap(input: Mat) {
        var bmp: Bitmap? = null
        val rgb = Mat()
        Imgproc.cvtColor(input, rgb, Imgproc.COLOR_BGR2RGB)
        try {
            bmp = Bitmap.createBitmap(rgb.cols(), rgb.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(rgb, bmp)
        } catch (e: CvException) {
            println(e.message)
        }

        val byteArrayOutputStream = ByteArrayOutputStream()
        bmp?.compress(Bitmap.CompressFormat.PNG, 100, byteArrayOutputStream)
        val byteArray: ByteArray = byteArrayOutputStream.toByteArray()
        val string = Base64.encodeToString(byteArray, Base64.NO_WRAP)
        println(string)
    }
}
