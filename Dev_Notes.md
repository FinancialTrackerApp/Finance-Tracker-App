#Backend
- to run the backend server be in the main folder and run uvicorn backend.app.app:app --reload
- Database Structure:-
    Table name - expenses
    sample table structure
    # | id | date       | category  | amount | text                       |
    # | -- | ---------- | --------- | ------ | -------------------------  |
    # | 1  | 2025-09-04 | Food      | 500.0  | spent 500 at KFC           |
    # | 2  | 2025-09-04 | Education | 5000.0 | paid 5000 for tuition fee  |
    # | 3  | 2025-09-05 | Transport | 200.0  | spent 200 on bus           |
#Frontend
- theres three main components to adding notes, namely saveNote(), deleteNoteAt(),  and fetchNotes()
- essentially, fetchnotes syncs the frontend and backend notes list. we call this function inside savenote/deletenoteat after the main functionality of saveNote and deleteNoteAt finishes, to update the index and id attributes.
- index , id attributes are necessary to sync frontend and backend. id is the id of the row of the note in the database, whereas index is what the frontend uses to track each note in ListView.
- basically, syncing ID and index is vital to keep functionality intact, which is why we need fetchnotes.
- savenotes now calls predict endpoint on press, original predict function is commented out for later use.
- total is updated by awaiting response data after delete/save(add+predict) functionality 