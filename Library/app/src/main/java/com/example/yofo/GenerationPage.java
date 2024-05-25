package com.example.yofo;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Environment;

import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

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

import com.github.barteksc.pdfviewer.PDFView;

public class GenerationPage extends AppCompatActivity {
    private static final String url = "http://192.168.1.44:5000";
    private static final int a5Width200dpi = 1169;
    private static final int a5Height200dpi = 1654;
    private static final int a5NumberOfSymbols = 35;
    private static final int a4Width200dpi = 1654;
    private static final int a4Height200dpi = 2339;
    private static final int a4NumberOfSymbols = 50;
    private static InputStream inputStream;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getSupportActionBar().hide();
        setContentView(R.layout.generated_result);

        Intent previousIntent = getIntent();
        final String text = (String) previousIntent.getExtras().get("text");
        final File fontFile = (File) previousIntent.getExtras().get("file");
        final String PDFormat = (String) previousIntent.getExtras().get("format");
        final String fileName = previousIntent.getExtras().get("name") + ".pdf";

        final int currentWidth = (PDFormat.equals("A4") ? a4Width200dpi : a5Width200dpi);
        final int currentHeight = (PDFormat.equals("A4") ? a4Height200dpi : a5Height200dpi);
        final int currentNumberOfSymbols = (PDFormat.equals("A4") ? a4NumberOfSymbols : a5NumberOfSymbols);

        PdfDocument document = null;
        try {
            File file = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS), fileName);
            document = new PdfDocument(new PdfWriter(file.getAbsolutePath()));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        assert document != null;
        PdfPage page = document.addNewPage(new PageSize(currentWidth, currentHeight));
        PdfCanvas canvas = new PdfCanvas(page);

        Thread sendGenerationRequest = new Thread(() -> {
            RequestBody requestBody = new MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart("text", text)
                    .addFormDataPart("number_of_symbols", String.valueOf(currentNumberOfSymbols))
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
        Thread thread = new Thread(() -> {
            ByteArrayOutputStream stream = new ByteArrayOutputStream();
            gotImage.compress(Bitmap.CompressFormat.JPEG, 100, stream);
            byte[] bitmapData = stream.toByteArray();
            ImageData imageData = ImageDataFactory.create(bitmapData);
            canvas.addImage(imageData, 0, gotImage.getHeight(), gotImage.getWidth(), false);
        });
        thread.start();
        try {
            thread.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        TextView fileNameTextView = findViewById(R.id.fileName);
        fileNameTextView.setText(fileName);
        PDFView pdfView = findViewById(R.id.pdfView);
        pdfView.setMinimumHeight((int) (1.4141 * pdfView.getWidth()));
        pdfView.fromFile(new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS), fileName))
                .fitEachPage(true)
                .load();

        document.close();

        ImageButton remove = findViewById(R.id.removeFile);
        remove.setOnClickListener(v -> {
            File file = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS), fileName);
            if (file.delete()) {
                Toast.makeText(this, "Successfully removed!", Toast.LENGTH_SHORT).show();
                Intent intent = new Intent(this, MainActivity.class);
                startActivity(intent);
            } else {
                Toast.makeText(this, "Something went wrong", Toast.LENGTH_SHORT).show();
            }
        });

    }
}
