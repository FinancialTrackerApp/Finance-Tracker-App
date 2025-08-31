import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (context) => MyAppState(),
      child: MaterialApp(
        title: 'Finance Tracker App',
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        ),
        home: MyHomePage(),
      ),
    );
  }
}

// App state
class Note {
  String text;
  Note({required this.text});
}

class MyAppState extends ChangeNotifier {
  String userInput = "";
  Map<String, dynamic> totals = {};
  bool isLoading = false;
  int? editingIndex;
  List<Note> notes = [];
  bool isEditing = false;
  String currentText = "";

  void updateInput(String newText) {
    userInput = newText;
    notifyListeners();
  }

  Future<void> predict() async {
    if (userInput.isEmpty) return;
    isLoading = true;
    notifyListeners();
    try {
      final url = Uri.parse("http://127.0.0.1:5000/predict");
      final response = await http.post(
        url,
        headers: {"Content-Type": "application/json"},
        body: json.encode({"text": userInput}),
      );
      if (response.statusCode == 200) {
        totals = json.decode(response.body);
      } else {
        print("Error: ${response.statusCode}");
      }
    } catch (e) { 
      print("Exception: $e");
    }
    isLoading = false;
    notifyListeners();
  }

  // Notes logic
  void startEditing({String text = "", int? index}) {
    currentText = text;
    isEditing = true;
    editingIndex = index; // track which note is being edited
    notifyListeners();
  }


  void stopEditing() {
    isEditing = false;
    currentText = "";
    notifyListeners();
  }
  void saveNote() {
    if (currentText.trim().isEmpty) {
      stopEditing();
      return;
    }
    if (editingIndex != null) {
      notes[editingIndex!] = Note(text: currentText); // update existing
    } else {
      notes.add(Note(text: currentText)); // add new
    }
    stopEditing();
  }

  void updateText(String text) {
    currentText = text;
    notifyListeners();
  }

  void addNote(String text) {
    notes.add(Note(text: text));
    notifyListeners();
  }
  void deleteNoteAt(int index) {
    notes.removeAt(index);
    notifyListeners();
  }
}

// Main page with NavigationRail
class MyHomePage extends StatefulWidget {
  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  var selectedIndex = 0;

  @override
  Widget build(BuildContext context) {
    Widget page;
    switch (selectedIndex) {
      case 0:
        page = GeneratorPage();
        break;
      default:
        throw UnimplementedError('no widget for $selectedIndex');
    }

    return LayoutBuilder(builder: (context, constraints) {
      return Scaffold(
        body: Row(
          children: [
            SafeArea(
              child: NavigationRail(
                extended: constraints.maxWidth >= 600,
                destinations: [
                  NavigationRailDestination(
                    icon: Icon(Icons.home),
                    label: Text('Home'),
                  )
                ],
                selectedIndex: selectedIndex,
                onDestinationSelected: (value) {
                  setState(() {
                    selectedIndex = value;
                  });
                },
              ),
            ),
            Expanded(
              child: Container(
                color: Theme.of(context).colorScheme.primaryContainer,
                child: page,
              ),
            ),
          ],
        ),
      );
    });
  }
}

// Generator page with input, buttons, totals, and notes
class GeneratorPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final appState = context.watch<MyAppState>();
    final size = MediaQuery.of(context).size;
    return Stack(
      children: [
        Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Input TextField
              Spacer(),
              TextField(
                onChanged: (text) => appState.updateInput(text),
                decoration: InputDecoration(
                  labelText: "Enter expense text",
                  border: OutlineInputBorder(),
                ),
              ),
              SizedBox(height: 12),
              // Buttons
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                mainAxisSize: MainAxisSize.min,
                children: [
                  ElevatedButton.icon(
                    onPressed: () => appState.predict(),
                    icon: Icon(Icons.arrow_right),
                    label: Text('Predict'),
                  ),
                  SizedBox(width: 10),
                  ElevatedButton.icon(
                    onPressed: () => appState.startEditing(),
                    icon: Icon(Icons.edit),
                    label: Text('Create Note'),
                  ),
                ],
              ),
              SizedBox(height: 20),
              // Totals box
              Align(
                alignment: Alignment.center, // align container to left
                child: IntrinsicWidth( // shrink container to fit text
                  child: Container(
                    padding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                    decoration: BoxDecoration(
                      border: Border.all(color: Colors.black54),
                      borderRadius: BorderRadius.circular(8),
                      color: Colors.white,
                    ),
                    child: appState.isLoading
                        ? Center(child: CircularProgressIndicator())
                        : appState.totals.isEmpty
                        ? Text("Totals will appear here", textAlign: TextAlign.left)
                        : Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: appState.totals.entries
                          .map((e) => Text("${e.key}: ₹${e.value}"))
                          .toList(),
                    ),
                  ),
                ),
              ),
              SizedBox(height: 12),
              // Notes list
              Expanded(
                child: ListView(
                  children: appState.notes
                      .asMap()
                      .entries
                      .map((entry) {
                    int idx = entry.key;
                    Note note = entry.value;
                    return GestureDetector(
                      onTap: () {
                        appState.startEditing(text: note.text, index: idx);
                      },
                      child: Container(
                        margin: EdgeInsets.symmetric(vertical: 6),
                        padding: EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: Colors.yellow[100],
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Expanded(
                              child: Text(
                                note.text,
                                maxLines: 5,
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                            IconButton(
                              icon: Icon(Icons.delete, color: Colors.red),
                              onPressed: () => appState.deleteNoteAt(idx),
                            ),
                          ],
                        ),
                      ),
                    );
                  })
                      .toList(),
                ),
              )
            ],
          ),
        ),
        // Animated Note Editor Overlay
        if (appState.isEditing) AnimatedNoteEditor(),
      ],
    );
  }
}

// Animated note editor
class AnimatedNoteEditor extends StatefulWidget {
  @override
  State<AnimatedNoteEditor> createState() => _AnimatedNoteEditorState();
}

class _AnimatedNoteEditorState extends State<AnimatedNoteEditor> {
  bool isExpanded = false;

  @override
  void initState() {
    super.initState();
    // Expand after a short delay
    Future.delayed(Duration(milliseconds: 50), () {
      if (mounted) {
        setState(() {
          isExpanded = true;
        });
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final appState = context.watch<MyAppState>();
    final screenSize = MediaQuery.of(context).size;

    // Animation values
    final topFactor = isExpanded ? 0.15 : 0.75;    // Fraction from top
    final heightFactor = isExpanded ? 0.55 : 0.15; // Fraction of screen height
    final widthFactor = isExpanded ? 0.5 : 0.9;   // Fraction of screen width

    return Stack(
      children: [
        // Semi-transparent background
        GestureDetector(
          onTap: () => appState.stopEditing(),
          child: Container(
            color: Colors.black54,
          ),
        ),


        AnimatedAlign(
          duration: Duration(milliseconds: 400),
          curve: Curves.easeInOut,
          alignment: Alignment(0, -1 + 2 * topFactor), // x=0 for horizontal center, y mapped to topFactor
          child: FractionallySizedBox(
            widthFactor: widthFactor,
            heightFactor: heightFactor,
            child: Material(
              borderRadius: BorderRadius.circular(16),
              color: Colors.white,
              child: Padding(
                padding: EdgeInsets.all(16),
                child: Column(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: TextEditingController(text: appState.currentText)
                          ..selection = TextSelection.fromPosition(
                            TextPosition(offset: appState.currentText.length),
                          ),
                        maxLines: null,
                        expands: true,
                        onChanged: (text) => appState.updateText(text),
                        decoration: InputDecoration(
                          hintText: "Write your note...",
                          border: InputBorder.none,
                        ),
                      ),
                    ),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        TextButton(
                          onPressed: () => appState.stopEditing(),
                          child: Text("Cancel"),
                        ),
                        ElevatedButton(
                          onPressed: () {
                            if (appState.currentText.trim().isNotEmpty) {
                              appState.saveNote();
                            }
                            appState.stopEditing();
                          },
                          child: Text("Save"),
                        ),
                      ],
                    )
                  ],
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }
}
