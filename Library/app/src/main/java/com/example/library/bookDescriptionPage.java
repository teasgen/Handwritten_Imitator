package com.example.library;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.io.IOException;
import java.util.Objects;

public class bookDescriptionPage extends AppCompatActivity {
    @SuppressLint("SetTextI18n")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.about_book);
        Objects.requireNonNull(getSupportActionBar()).hide();
        TextView authorView = findViewById(R.id.authorAbout);
        TextView titleView = findViewById(R.id.labelAbout);
        TextView descriptionView = findViewById(R.id.bookAbout);
        ImageView imageView = findViewById(R.id.imageAbout);

        NotesDataset mDatabase = NotesDataset.getInstance(this);
        Intent prevIntent = getIntent();
        Note note = mDatabase.listNotes().get(Integer.parseInt(prevIntent.getStringExtra("position")));

        String author = note.getAuthor();
        String title = note.getTitle();
        String description = note.getDescription();
        Uri imgUri = Uri.parse("file://" + note.getImgPath());
        Bitmap bitmap = null;
        try {
            bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imgUri);
        } catch (IOException e) {
            e.printStackTrace();
        }

        imageView.setImageBitmap(bitmap);
        authorView.setText("@" + author);
        titleView.setText(title);
        descriptionView.setText(description);
    }
}
