import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:http/http.dart' as http;
import 'package:intl/intl.dart';
import 'package:google_fonts/google_fonts.dart';
import 'dart:async';
import 'package:fl_chart/fl_chart.dart';

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

  Future<void> predict([String? text]) async {
    final input = text ?? userInput; // Use provided text OR fallback to userInput
    if (input.isEmpty) return;

    isLoading = true;
    notifyListeners();

    try {
      // Use current date in ISO format
      final noww = DateTime.now();
      final now = DateFormat('dd-MM-yyyy').format(noww);

      final url = Uri.parse("http://127.0.0.1:8000/predict"); // adjust if using physical device
      final response = await http.post(
        url,
        headers: {"Content-Type": "application/json"},
        body: json.encode({
          "text": input,
          "date": now, // <-- send the date field
        }),
      );

      if (response.statusCode == 200) {
        totals = json.decode(response.body); // update totals box
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
    final noteText = currentText.trim();

    if (noteText.isEmpty) {
      stopEditing();
      return;
    }

    // Save or update note
    if (editingIndex != null) {
      notes[editingIndex!] = Note(text: noteText);
    } else {
      notes.add(Note(text: noteText));
    }

    // Call predict() with the saved note's text
    predict(noteText);

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
      case 1:
        page= GraphPage();
        break;
      default:
        throw UnimplementedError('no widget for $selectedIndex');
    }

    return Scaffold(
      appBar: AppBar(
        title: Text("Expense Tracker",
          style: GoogleFonts.lato(
            fontSize: 20,
          ),
        ),

        leading: Builder(
          builder: (context) => IconButton(
            icon: Icon(Icons.menu), // hamburger menu
            onPressed: () => Scaffold.of(context).openDrawer(),
          ),
        ),
      ),
      drawer: Drawer(
        child: ListView(
          children: [
            ListTile(
              leading: Icon(Icons.home),
              title: Text("Home"),
              onTap: () {
                setState(() {
                  selectedIndex = 0;
                });
                Navigator.of(context).pop(); // close drawer
              },
            ),
            ListTile(
              leading: Icon(Icons.auto_graph),
              title: Text("Expenditure Graph"),
              onTap: () {
                setState(() {
                  selectedIndex = 1;
                });
                Navigator.of(context).pop(); // close drawer
              },
            ),
            // Add more items here
          ],
        ),
      ),
      body: Container(
        color: Theme.of(context).colorScheme.primaryContainer,
        child: page,
      ),
    );
  }
}

// Generator page with input, buttons, totals, and notes
class GeneratorPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final appState = context.watch<MyAppState>();
    final size = MediaQuery.of(context).size;

    return Scaffold(
      backgroundColor: Theme.of(context).colorScheme.primaryContainer, // ✅ Back to light blue
      body: Stack(
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                DateTimeDisplayWidget(),
                SizedBox(height: 20),

                // Totals box
                Align(
                  alignment: Alignment.center,
                  child: IntrinsicWidth(
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                      decoration: BoxDecoration(
                        border: Border.all(color: Colors.black54),
                        borderRadius: BorderRadius.circular(8),
                        color: Colors.white, // ✅ Totals box stays white
                      ),
                      child: appState.isLoading
                          ? const Center(child: CircularProgressIndicator())
                          : appState.totals.isEmpty
                          ? const Text("Today's Totals:", textAlign: TextAlign.left)
                          : Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: appState.totals.entries
                            .map((e) => Text("${e.key}: ₹${e.value}"))
                            .toList(),
                      ),
                    ),
                  ),
                ),

                const SizedBox(height: 12),

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
                          margin: const EdgeInsets.symmetric(vertical: 6),
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: Colors.blue[50],
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
                                icon: const Icon(Icons.delete, color: Colors.red),
                                onPressed: () => appState.deleteNoteAt(idx),
                              ),
                            ],
                          ),
                        ),
                      );
                    }).toList(),
                  ),
                ),
              ],
            ),
          ),

          // Animated Note Editor Overlay
          if (appState.isEditing) AnimatedNoteEditor(),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () => appState.startEditing(),
        child: const Icon(Icons.edit, color: Colors.white),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.endFloat,
    );
  }
}
class GraphPage extends StatelessWidget {
  final List<Map<String, dynamic>> monthlyData = [
    {"month": "Jan", "total": 5000},
    {"month": "Feb", "total": 6000},
    {"month": "Mar", "total": 5500},
    {"month": "Apr", "total": 7000},
    {"month": "May", "total": 6500},
    {"month": "Jun", "total": 7200},

  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).colorScheme.primaryContainer,
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: LineChart(
          LineChartData(
            lineBarsData: [
              LineChartBarData(
                spots: monthlyData.asMap().entries.map((entry) {
                  final idx = entry.key;
                  final data = entry.value;
                  return FlSpot(idx.toDouble(), data['total'].toDouble());
                }).toList(),
                isCurved: true,
                barWidth: 3,
                dotData: FlDotData(show: true),
                color: Colors.blue,
              ),
            ],
            titlesData: FlTitlesData(
              bottomTitles: AxisTitles(
                sideTitles: SideTitles(
                  showTitles: true,
                  getTitlesWidget: (value, meta) {
                    final idx = value.toInt();
                    if (idx < 0 || idx >= monthlyData.length) return Container();
                    return Text(monthlyData[idx]['month']);
                  },
                  reservedSize: 30,
                ),
              ),
              leftTitles: AxisTitles(
                sideTitles: SideTitles(showTitles: true, reservedSize: 40),
              ),
            ),
            gridData: FlGridData(show: true),
            lineTouchData: LineTouchData(
              enabled: true,
              touchTooltipData: LineTouchTooltipData(
                getTooltipItems: (touchedSpots) {
                  return touchedSpots.map((spot) {
                    final idx = spot.spotIndex;
                    return LineTooltipItem(
                      "${monthlyData[idx]['month']}\n₹${spot.y.toStringAsFixed(0)}",
                      const TextStyle(color: Colors.white),
                    );
                  }).toList();
                },
              ),
            ),
          ),
        )
      ),
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
class DateTimeDisplayWidget extends StatefulWidget {
  const DateTimeDisplayWidget({Key? key}) : super(key: key);
  @override
  State<DateTimeDisplayWidget> createState() => _DateTimeDisplayWidgetState();
}

class _DateTimeDisplayWidgetState extends State<DateTimeDisplayWidget> {
  late String formattedDateTime;
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _updateDateTime();
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (mounted) {
        setState(() => _updateDateTime());
      }
    });
  }

  void _updateDateTime() {
    DateTime now = DateTime.now();

    // Format weekday, day with suffix, and month
    String weekday = DateFormat('EEEE').format(now);
    String month = DateFormat('MMMM').format(now);
    int day = now.day;

    String suffix;
    if (day >= 11 && day <= 13) {
      suffix = "th";
    } else {
      switch (day % 10) {
        case 1:
          suffix = "st";
          break;
        case 2:
          suffix = "nd";
          break;
        case 3:
          suffix = "rd";
          break;
        default:
          suffix = "th";
      }
    }

    formattedDateTime = "$weekday, $day$suffix $month";
  }

  @override
  void dispose() {
    _timer?.cancel(); // ✅ Stop the timer when widget is removed
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Text(
      formattedDateTime,
      style: GoogleFonts.lato(
        fontSize: 20,
        fontWeight: FontWeight.bold,
      ),
    );
  }
}


