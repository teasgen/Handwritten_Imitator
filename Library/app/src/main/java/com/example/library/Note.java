package com.example.library;

public class Note {
    private int id;
    private String author;
    private String title;
    private String description;
    private String imgPath;

    Note(String imgPath, String author, String title, String description) {
        this.imgPath = imgPath;
        this.author = author;
        this.title = title;
        this.description = description;
    }

    Note(int id, String imgPath, String author, String title, String description) {
        this.imgPath = imgPath;
        this.id = id;
        this.author = author;
        this.title = title;
        this.description = description;
    }

    public String getImgPath() {
        return imgPath;
    }

    public int getId() {
        return id;
    }

    public String getAuthor() {
        return author;
    }

    public String getTitle() {
        return title;
    }

    public String getDescription() {
        return description;
    }

    public void setImgPath(String imgPath) {
        this.imgPath = imgPath;
    }

    public void setAuthor(String author) {
        this.author = author;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public void setId(int id) {
        this.id = id;
    }

    public void setTitle(String title) {
        this.title = title;
    }
}
