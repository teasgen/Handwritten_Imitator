package com.example.yofo;

import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import java.util.Objects;

public class MakingNote extends AppCompatActivity {
    private static final int IMAGEPICK_GALLERY_REQUEST = 300;
    String[] storagePermission;
    String imgPath;
    NotesDataset mDatabase;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.note_creating);
        Objects.requireNonNull(getSupportActionBar()).hide();
        Button pageUploading = findViewById(R.id.uploadImage);
        storagePermission = new String[]{android.Manifest.permission.WRITE_EXTERNAL_STORAGE};
        pageUploading.setOnClickListener(view -> {
            Intent intent = new Intent(Intent.ACTION_PICK,
                    android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            startActivityForResult(intent, IMAGEPICK_GALLERY_REQUEST);
        });
        mDatabase = NotesDataset.getInstance(this);
        Button save = findViewById(R.id.saveButton);
        EditText title = findViewById(R.id.editBookLabel);

        Intent previousIntent = getIntent();
        int position = Integer.parseInt(previousIntent.getStringExtra("position"));
        if (position >= 0) {
            Note note = mDatabase.listNotes().get(position);
            title.setText(note.getTitle());
            imgPath = note.getImgPath();
        }

        save.setOnClickListener(view -> {
            if (title.getText().length() >= 25)
                Toast.makeText(this.getApplicationContext(), "Название слишком длинное!", Toast.LENGTH_LONG).show();
            else if (title.getText().length() == 0)
                Toast.makeText(this.getApplicationContext(), "Введи название!", Toast.LENGTH_LONG).show();
            else if (imgPath == null || imgPath.length() == 0)
                Toast.makeText(this.getApplicationContext(), "Укажи фотографию!", Toast.LENGTH_LONG).show();
            else {
                if (position >= 0)
                    mDatabase.deleteNote(mDatabase.listNotes().get(position).getId());
                mDatabase.addNote(new Note(
                        imgPath,
                        title.getText().toString()));



                Intent intent = new Intent(MakingNote.this,
                        MainActivity.class);
                startActivity(intent);
            }
        });
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        assert data != null;
        Uri uri = data.getData();
        String[] projection = {MediaStore.Images.Media.DATA};
        Cursor cursor = getContentResolver().query(uri, projection, null, null, null);
        int column_index = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
        cursor.moveToFirst();
        imgPath = cursor.getString(column_index);
        Toast.makeText(this.getApplicationContext(), imgPath, Toast.LENGTH_LONG).show();
    }
}
