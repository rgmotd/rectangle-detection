package jp.co.smartbank.rectangledetector.sample.ui

import android.graphics.Point
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import jp.co.smartbank.rectangledetector.dto.DetectionResult
import jp.co.smartbank.rectangledetector.dto.Rectangle
import kotlin.math.roundToInt

@Composable
fun DetectedRectangleLayer(
    modifier: Modifier = Modifier,
    previewSize: Size = Size.Zero,
    detectionResult: DetectionResult? = null
) {
    if (detectionResult == null || previewSize == Size.Zero) {
        return
    }

    Canvas(modifier = modifier) {
        detectionResult.rectanglesInPreview(previewSize).also {
            val path = Path().apply {
                moveTo(it.topLeft.x.toFloat(), it.topLeft.y.toFloat())
                lineTo(it.topRight.x.toFloat(), it.topRight.y.toFloat())
                lineTo(it.bottomRight.x.toFloat(), it.bottomRight.y.toFloat())
                lineTo(it.bottomLeft.x.toFloat(), it.bottomLeft.y.toFloat())
                close()
            }
            drawPath(
                path = path,
                color = Color.Red.copy(alpha = 0.5f),
                style = Stroke(width = 4.dp.toPx())
            )
        }
    }
}

private fun DetectionResult.previewRectInImage(previewSize: Size): Rect {
    val widthRatio = previewSize.width / imageSize.width
    val heightRatio = previewSize.height / imageSize.height
    return if (widthRatio > heightRatio) {
        val previewHeightInImage = (previewSize.height / widthRatio)
        Rect(
            Offset(0f, (imageSize.height - previewHeightInImage) / 2),
            Size(imageSize.width.toFloat(), previewHeightInImage)
        )
    } else {
        val previewWidthInImage = (previewSize.width / heightRatio)
        Rect(
            Offset((imageSize.width - previewWidthInImage) / 2, 0f),
            Size(previewWidthInImage, imageSize.height.toFloat())
        )
    }
}

private fun DetectionResult.rectanglesInPreview(previewSize: Size): Rectangle {
    val previewRectInImage = previewRectInImage(previewSize)
    val scaleRatio = previewSize.width / previewRectInImage.width
    return Rectangle(
        topLeft = Point(
            ((rectangle?.topLeft?.x?.minus(previewRectInImage.left))?.times(scaleRatio))?.roundToInt() ?: 0,
            ((rectangle?.topLeft?.y?.minus(previewRectInImage.top))?.times(scaleRatio))?.roundToInt() ?: 0
        ),
        topRight = Point(
            ((rectangle?.topRight?.x?.minus(previewRectInImage.left))?.times(scaleRatio))?.roundToInt() ?: 0,
            ((rectangle?.topRight?.y?.minus(previewRectInImage.top))?.times(scaleRatio))?.roundToInt() ?: 0
        ),
        bottomLeft = Point(
            ((rectangle?.bottomLeft?.x?.minus(previewRectInImage.left))?.times(scaleRatio))?.roundToInt() ?: 0,
            ((rectangle?.bottomLeft?.y?.minus(previewRectInImage.top))?.times(scaleRatio))?.roundToInt() ?: 0
        ),
        bottomRight = Point(
            ((rectangle?.bottomRight?.x?.minus(previewRectInImage.left))?.times(scaleRatio))?.roundToInt() ?: 0,
            ((rectangle?.bottomRight?.y?.minus(previewRectInImage.top))?.times(scaleRatio))?.roundToInt() ?: 0
        )
    )
}
