package com.example.yofo;

import static java.lang.Math.min;

import android.annotation.SuppressLint;
import android.content.Intent;
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

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class GenerationPage extends AppCompatActivity {
    private static final String url = "http://192.168.1.44:5000";
    private static final int p5Width200dpi = 1169;
    private static final int p5Height200dpi = 1654;
    private static final int p5NumberOfSymbols = 35;
    private static InputStream inputStream;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getSupportActionBar().hide();
        setContentView(R.layout.generated_result);

        Intent previousIntent = getIntent();
        final String text = (String) previousIntent.getExtras().get("text");
        final File fontFile = (File) previousIntent.getExtras().get("file");

        PdfDocument document = null;
        try {
            File file = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS), "output.pdf");
            document = new PdfDocument(new PdfWriter(file.getAbsolutePath()));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        assert document != null;
        // PdfPage page = document.addNewPage(new PageSize(gotImage.getWidth(), gotImage.getHeight()));
        PdfPage page = document.addNewPage(new PageSize(p5Width200dpi, p5Height200dpi));
        PdfCanvas canvas = new PdfCanvas(page);

        for (int i = 0; i < text.length(); i += p5NumberOfSymbols) {
            String currentText = text.substring(i, min(i + p5NumberOfSymbols, text.length()));
            Thread sendGenerationRequest = new Thread(() -> {
                RequestBody requestBody = new MultipartBody.Builder()
                        .setType(MultipartBody.FORM)
                        .addFormDataPart("text", currentText)
                        .addFormDataPart("font", fontFile.getName(), RequestBody.create(MediaType.parse("image/jpeg"), fontFile))
                        .build();

                Request request = new Request.Builder()
                        .url(url + "/upload")
                        .post(requestBody)
                        .build();

                OkHttpClient client = new OkHttpClient();
                Response response = null;
                try {
                    response = client.newCall(request).execute();
                } catch (IOException e) {
                    e.printStackTrace();
                }

                byte[] imageData = new byte[0];
                try {
                    assert response != null;
                    imageData = response.body().bytes();
                } catch (IOException e) {
                    e.printStackTrace();
                }

                inputStream = new ByteArrayInputStream(imageData);
            });
            sendGenerationRequest.start();
            try {
                sendGenerationRequest.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            Bitmap gotImage = BitmapFactory.decodeStream(inputStream);
            int finalI = i;
            Thread thread = new Thread(() -> {
                ByteArrayOutputStream stream = new ByteArrayOutputStream();
                gotImage.compress(Bitmap.CompressFormat.JPEG, 100, stream);
                byte[] bitmapData = stream.toByteArray();
                ImageData imageData = ImageDataFactory.create(bitmapData);
                canvas.addImage(imageData, 0, p5Height200dpi - 64 * ((int)(finalI / p5NumberOfSymbols + 1)), gotImage.getWidth(), false);
            });
            thread.start();
            try {
                thread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            ImageView imageView = findViewById(R.id.generated);
            imageView.setImageBitmap(gotImage);
        }
        document.close();
    }
}
/*
new Thread(() -> {
    File file = new File(note.getImgPath());
    EditText editText = findViewById(R.id.addText);
    String text = String.valueOf(editText.getText());
    RequestBody requestBody = new MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("text", text)
            .addFormDataPart("font", file.getName(), RequestBody.create(MediaType.parse("image/jpeg"), file))
            .build();

    Request request = new Request.Builder()
            .url(url + "/upload")
            .post(requestBody)
            .build();

    OkHttpClient client = new OkHttpClient();
    Response response = null;
    try {
        response = client.newCall(request).execute();
    } catch (IOException e) {
        e.printStackTrace();
    }

    byte[] imageData = new byte[0];
    try {
        assert response != null;
        imageData = response.body().bytes();
    } catch (IOException e) {
        e.printStackTrace();
    }

    inputStream = new ByteArrayInputStream(imageData);
    Intent intent = new Intent(this, GenerationPage.class);
    startActivity(intent);
}).start();
 */

