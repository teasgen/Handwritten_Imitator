package com.example.yofo;

import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Environment;
import android.widget.ImageView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.itextpdf.io.image.ImageData;
import com.itextpdf.io.image.ImageDataFactory;
import com.itextpdf.kernel.geom.PageSize;
import com.itextpdf.kernel.pdf.PdfDocument;
import com.itextpdf.kernel.pdf.PdfPage;
import com.itextpdf.kernel.pdf.PdfWriter;
import com.itextpdf.kernel.pdf.canvas.PdfCanvas;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.InputStream;

public class GenerationPage extends AppCompatActivity {
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getSupportActionBar().hide();
        setContentView(R.layout.generated_result);

        InputStream inputStream = WriteTextPage.getInputStream();
        Bitmap gotImage = BitmapFactory.decodeStream(inputStream);

        PdfDocument document = null;
        try {
            File file = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS), "output.pdf");
            document = new PdfDocument(new PdfWriter(file.getAbsolutePath()));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        assert document != null;
        // PdfPage page = document.addNewPage(new PageSize(gotImage.getWidth(), gotImage.getHeight()));
        PdfPage page = document.addNewPage(new PageSize(1169, 1654));
        PdfCanvas canvas = new PdfCanvas(page);

        Thread thread = new Thread(() -> {
            ByteArrayOutputStream stream = new ByteArrayOutputStream();
            gotImage.compress(Bitmap.CompressFormat.JPEG, 100, stream);
            byte[] bitmapData = stream.toByteArray();
            ImageData imageData = ImageDataFactory.create(bitmapData);
            canvas.addImage(imageData, 0, 1654 - 64, gotImage.getWidth(), false);
        });
        thread.start();
        try {
            thread.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        document.close();
        ImageView imageView = findViewById(R.id.generated);
        imageView.setImageBitmap(gotImage);
    }
}

