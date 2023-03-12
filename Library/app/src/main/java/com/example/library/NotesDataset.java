package com.example.library;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import java.util.ArrayList;

public class NotesDataset extends SQLiteOpenHelper {
    private static NotesDataset sInstance;

    private static final int DATABASE_VERSION = 1;
    private static final String DATABASE_NAME = "Notes";
    private static final String TABLE_CONTACTS = "Notes";
    private static final String COLUMN_ID = "_id";
    private static final String COLUMN_AUTHOR = "author";
    private static final String COLUMN_PATH = "imgPath";
    private static final String COLUMN_TITLE = "title";
    private static final String COLUMN_DESCRIPTION = "description";

    public static synchronized NotesDataset getInstance(Context context) {
        if (sInstance == null) {
            sInstance = new NotesDataset(context.getApplicationContext());
        }
        return sInstance;
    }

    public NotesDataset(Context context) {
        super(context, DATABASE_NAME, null, DATABASE_VERSION);
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        String createTable = "CREATE TABLE "
                + TABLE_CONTACTS + "(" + COLUMN_ID
                + " INTEGER PRIMARY KEY,"
                + COLUMN_PATH + " TEXT,"
                + COLUMN_AUTHOR + " TEXT,"
                + COLUMN_TITLE + " TEXT,"
                + COLUMN_DESCRIPTION + " TEXT" + ")";
        db.execSQL(createTable);
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        db.execSQL("DROP TABLE IF EXISTS " + TABLE_CONTACTS);
        onCreate(db);
    }

    ArrayList<Note> listNotes() {
        String sql = "select * from " + TABLE_CONTACTS;
        SQLiteDatabase db = this.getReadableDatabase();
        ArrayList<Note> storeNotes = new ArrayList<>();
        Cursor cursor = db.rawQuery(sql, null);
        if (cursor.moveToFirst()) {
            do {
                int id = Integer.parseInt(cursor.getString(0));
                String path = cursor.getString(1);
                String author = cursor.getString(2);
                String title = cursor.getString(3);
                String desc = cursor.getString(4);
                storeNotes.add(new Note(id, path, author, title, desc));
            } while (cursor.moveToNext());
        }
        cursor.close();
        return storeNotes;
    }

    void addNote(Note note) {
        ContentValues values = new ContentValues();
        values.put(COLUMN_PATH, note.getImgPath());
        values.put(COLUMN_AUTHOR, note.getAuthor());
        values.put(COLUMN_TITLE, note.getTitle());
        values.put(COLUMN_DESCRIPTION, note.getDescription());
        SQLiteDatabase db = this.getWritableDatabase();
        db.insert(TABLE_CONTACTS, null, values);
    }

    void updateNote(Note note) {
        ContentValues values = new ContentValues();
        values.put(COLUMN_PATH, note.getImgPath());
        values.put(COLUMN_AUTHOR, note.getAuthor());
        values.put(COLUMN_TITLE, note.getTitle());
        values.put(COLUMN_DESCRIPTION, note.getDescription());
        SQLiteDatabase db = this.getWritableDatabase();
        db.update(TABLE_CONTACTS, values, COLUMN_ID + " = ?", new String[]{String.valueOf(note.getId())});
    }

    void deleteNote(int id) {
        SQLiteDatabase db = this.getWritableDatabase();
        db.delete(TABLE_CONTACTS, COLUMN_ID + " = ?", new String[]{String.valueOf(id)});
    }

    void clear() {
        SQLiteDatabase db = this.getWritableDatabase();
        String sql = "delete from " + TABLE_CONTACTS;
        db.execSQL(sql);
    }
}
