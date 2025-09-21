import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:http/http.dart' as http;
import 'package:intl/intl.dart';
import 'package:google_fonts/google_fonts.dart';
import 'dart:async';
import 'package:fl_chart/fl_chart.dart';
import 'package:table_calendar/table_calendar.dart';


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
        title: 'FinanceKoi',
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        ),
        home: SplashScreen(), // start with splash

      ),
    );
  }
}

// App state
class Note {
  final int id;
  final String text;

  Note({required this.id, required this.text});
}

class MyAppState extends ChangeNotifier {
  String userInput = "";
  Map<String, dynamic> totals = {};
  bool isLoading = false;
  int? editingIndex;
  List<Note> notes = [];
  bool isEditing = false;
  String currentText = "";
  DateTime currentDate = DateTime.now();

  void setCurrentDate(DateTime newDate) {
    currentDate = newDate;
    notifyListeners();
  }

  void updateInput(String newText) {
    userInput = newText;
    notifyListeners();
  }

  // Future<void> predict([String? text]) async {
  //   final input = text ?? userInput; // Use provided text OR fallback to userInput
  //   if (input.isEmpty) return;
  //
  //   isLoading = true;
  //   notifyListeners();
  //
  //   try {
  //     // Use current date in ISO format
  //     final noww = DateTime.now();
  //     final now = DateFormat('dd-MM-yyyy').format(noww);
  //
  //     final url = Uri.parse("http://127.0.0.1:8000/predict"); // adjust if using physical device
  //     final response = await http.post(
  //       url,
  //       headers: {"Content-Type": "application/json"},
  //       body: json.encode({
  //         "text": input,
  //         "date": now, // <-- send the date field
  //       }),
  //     );
  //
  //     if (response.statusCode == 200) {
  //       totals = json.decode(response.body); // update totals box
  //     } else {
  //       print("Error: ${response.statusCode}");
  //     }
  //   } catch (e) {
  //     print("Exception: $e");
  //   }
  //
  //   isLoading = false;
  //   notifyListeners();
  // }

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

  Future<void> saveNote() async {
    final noteText = currentText.trim();
    final formattedDate = DateFormat('yyyy-MM-dd').format(currentDate);
    if (noteText.isEmpty) {
      stopEditing();
      return;
    }

    isLoading = true;
    notifyListeners();

    try {
      // Use the currentDate you're on, NOT DateTime.now()


      final url = Uri.parse("http://127.0.0.1:8000/predict");
      final response = await http.post(
        url,
        headers: {"Content-Type": "application/json"},
        body: json.encode({
          "text": noteText,
          "date": formattedDate,
        }),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);

        // Update totals immediately from backend
        if (data["Today's Total"] != null) {
          totals = {"Today's Total": data["Today's Total"]};
        }

        // Refresh notes for the currentDate (not today)
        await fetchNotes(date: formattedDate);

        print(
            "Saved successfully! Updated total: ${data["Today's Total"]} for date: $formattedDate");
      } else {
        print("Failed to save note: ${response.body}");
      }
    } catch (e) {
      print("Exception while saving note: $e");
    }

    isLoading = false;
    stopEditing();
    notifyListeners();
  }

  void updateText(String text) {
    currentText = text;
    notifyListeners();
  }


  Future<void> fetchNotes({required String date}) async {
    final url = Uri.parse("http://127.0.0.1:8000/expenses?date=$date");
    final response = await http.get(url);
    print("$date");
    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      notes = data.map<Note>((item) {
        return Note(id: item['id'], text: item['text']);
      }).toList();
      notifyListeners();
    } else {
      print("Failed to fetch notes: ${response.body}");
    }
  }

  Future<void> deleteNoteAt(int index) async {
    final appState = this; // assuming this is inside MyAppState
    final expenseId = notes[index].id;
    final url = Uri.parse('http://127.0.0.1:8000/expenses/$expenseId');

    try {
      final response = await http.delete(url);

      if (response.statusCode == 200) {
        final data = json.decode(response.body);

        // Refresh notes list
        final formattedDate = DateFormat('yyyy-MM-dd').format(currentDate);

        await fetchNotes(date: formattedDate);

        // Update totals using the backend's updated_total
        if (data['updated_total'] != null) {
          appState.totals = {"Today's Total": data['updated_total']};
        }

        notifyListeners();
        print("Deleted successfully! Updated total: ${data['updated_total']}");
      } else {
        print("Failed to delete note: ${response.body}");
      }
    } catch (e) {
      print("Exception while deleting note: $e");
    }
  }

  Future<void> uploadFile(BuildContext context) async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf', 'jpg', 'png', 'jpeg'],
    );

    if (result != null) {
      File file = File(result.files.single.path!);

      var request = http.MultipartRequest(
        'POST',
        Uri.parse('http://127.0.0.1:8000/parse_receipt'),
      );
      request.files.add(await http.MultipartFile.fromPath('file', file.path));

      var response = await request.send();

      if (response.statusCode == 200) {
        String respStr = await response.stream.bytesToString();
        var jsonData = json.decode(respStr);
        print("Parsed receipt data: $jsonData");

        // FIX: Check if jsonData['items'] is null, and provide an empty list if it is.
        // Also, ensure jsonData itself isn't null and is a Map.
        List<ReceiptItem> items = []; // Default to empty list
        if (jsonData != null && jsonData is Map && jsonData['items'] != null) {
          items = parseItems(jsonData['items'] as List<dynamic>);
        } else {
          // Optionally, handle the case where 'items' is missing or jsonData is not as expected.
          print("Warning: 'items' key not found in response or jsonData is not a Map.");
        }
        
        double total = jsonData != null && jsonData is Map && jsonData['total'] != null 
                       ? (jsonData['total'] as num).toDouble() 
                       : 0.0;

        // Use the context passed from the UI
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) => ReceiptPreviewScreen(items: items, total: total),
          ),
        );
      } else {
        print("Failed to parse receipt. Status: ${response.statusCode}");
      }
    } else {
      print("No file selected");
    }
  }

  List<ReceiptItem> parseItems(List<dynamic> jsonItems) {
    return jsonItems.map((e) => ReceiptItem.fromJson(e)).toList();
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
    String appBarTitle; // Variable to hold the AppBar title

    switch (selectedIndex) {
      case 0:
        page = GeneratorPage();
        appBarTitle = "Expense Tracker"; // Title for Home page
        break;
      case 1:
        page = GraphPage();
        appBarTitle = "Expenditure Graph"; // Title for Graph page
        break;
      case 2: // New case for Settings
        page = SettingsPage();
        appBarTitle = "Settings"; // Title for Settings page
        break;
      default:
        throw UnimplementedError('no widget for $selectedIndex');
    }

    return Scaffold(
      appBar: AppBar(
        title: Text(appBarTitle, // Use the dynamic title here
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
              leading: Icon(Icons.money),
              title: Text("Expense Tracker",
                  style: GoogleFonts.lato()
              ),
              onTap: () {
                setState(() {
                  selectedIndex = 0;
                });
                Navigator.of(context).pop(); // close drawer
              },
            ),
            ListTile(
              leading: Icon(Icons.auto_graph),
              title: Text("Expenditure Graph", style: GoogleFonts.lato()),
              onTap: () {
                setState(() {
                  selectedIndex = 1;
                });
                Navigator.of(context).pop(); // close drawer
              },
            ),
            ListTile( // New ListTile for Settings
              leading: Icon(Icons.settings),
              title: Text("Settings", style: GoogleFonts.lato()),
              onTap: () {
                setState(() {
                  selectedIndex = 2;
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
class GeneratorPage extends StatefulWidget {
  @override

  State<GeneratorPage> createState() => _GeneratorPageState();

}

class _GeneratorPageState extends State<GeneratorPage> {

  void goToPreviousDay() {
    final appState = context.read<MyAppState>();
    appState.setCurrentDate(appState.currentDate.subtract(const Duration(days: 1)));
    fetchNotesForCurrentDate();
  }

  void goToNextDay() {
    final appState = context.read<MyAppState>();

    // Only allow moving forward if the next day is today or earlier
    final nextDay = appState.currentDate.add(const Duration(days: 1));
    final today = DateTime.now();

    // Compare only the date part, ignore hours/minutes
    final nextDayDateOnly = DateTime(nextDay.year, nextDay.month, nextDay.day);
    final todayDateOnly = DateTime(today.year, today.month, today.day);

    if (nextDayDateOnly.isAfter(todayDateOnly)) {
      // Do nothing if nextDay is in the future
      return;
    }

    appState.setCurrentDate(nextDay);
    fetchNotesForCurrentDate();
  }

  void fetchNotesForCurrentDate() {
    final appState = context.read<MyAppState>();
    print("Fetching notes for: ${appState.currentDate}");
    final formattedDate = DateFormat('yyyy-MM-dd').format(appState.currentDate);

    appState.fetchNotes(date: formattedDate);
  }
  @override
  void initState() {
    super.initState();

    // Schedule fetch after widget is built so context is available
    Future.microtask(() {
      final appState = Provider.of<MyAppState>(context, listen: false);
      final formattedDate = DateFormat('yyyy-MM-dd').format(appState.currentDate);
      appState.fetchNotes(date: formattedDate);
    });
  }

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
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    IconButton(
                      icon: Icon(Icons.arrow_left),
                      onPressed: goToPreviousDay,
                    ),
                    DateTimeDisplayWidget(),
                    Builder(
                      builder: (context) {
                        final appState = context.watch<MyAppState>();
                        final nextDay = appState.currentDate.add(const Duration(days: 1));
                        final today = DateTime.now();
                        final nextDayDateOnly = DateTime(nextDay.year, nextDay.month, nextDay.day);
                        final todayDateOnly = DateTime(today.year, today.month, today.day);

                        final canGoNext = !nextDayDateOnly.isAfter(todayDateOnly);

                        return IconButton(
                          icon: Icon(Icons.arrow_right),
                          onPressed: canGoNext ? goToNextDay : null, // disabled if cannot go next
                        );
                      },
                    ),
                    DatePickerButton(),
                  ],
                ),

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
      floatingActionButton: Row(
        mainAxisAlignment: MainAxisAlignment.end, // Aligns buttons to the end of the Row
        children: <Widget>[
          FloatingActionButton(
            onPressed: () {
              appState.uploadFile(context);
              print("File upload pressed");
            },
            heroTag: null, // Add unique heroTag or set to null if not animating between screens
            child: const Icon(Icons.upload_file, color: Colors.white),
            tooltip: 'Upload File',
          ),
          SizedBox(width: 10), // Spacing between the buttons
          FloatingActionButton(
            onPressed: () => appState.startEditing(),
            heroTag: null, // Add unique heroTag or set to null
            child: const Icon(Icons.edit, color: Colors.white),
            tooltip: 'Edit Note',
          ),
        ],
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.endFloat,
    );
  }
}
Future<List<Map<String, dynamic>>> fetchDailyData() async {
  final response = await http.get(Uri.parse("http://127.0.0.1:8000/stats/daily"));
  if (response.statusCode == 200) {
    final List<dynamic> data = json.decode(response.body);
    return data.map((entry) {
      final date = entry['date'] as String;
      final total = (entry['total'] as num).toDouble();

      // Format label for x-axis
      final dateObj = DateTime.parse(date);
      final label = "${_monthShort(dateObj.month)} ${dateObj.day}";

      return {"date": date, "label": label, "total": total};
    }).toList();
  } else {
    throw Exception("Failed to load daily stats");
  }
}

String _monthShort(int month) {
  const months = [
    "Jan","Feb","Mar","Apr","May","Jun",
    "Jul","Aug","Sep","Oct","Nov","Dec"
  ];
  return months[month - 1];
}

// Fetch category breakdown for a date
Future<Map<String, double>> fetchCategoryBreakdown(String date) async {
  final response = await http.get(Uri.parse("http://127.0.0.1:8000/stats/day/$date"));
  if (response.statusCode == 200) {
    final Map<String, dynamic> data = json.decode(response.body);
    return data.map((key, value) => MapEntry(key, (value as num).toDouble()));
  } else {
    throw Exception("Failed to load category stats");
  }
}
class GraphPage extends StatelessWidget {
  const GraphPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).colorScheme.primaryContainer,
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: FutureBuilder<List<Map<String, dynamic>>>(
          future: fetchDailyData(),
          builder: (context, snapshot) {
            if (snapshot.connectionState == ConnectionState.waiting) {
              return const Center(child: CircularProgressIndicator());
            } else if (snapshot.hasError) {
              return Center(child: Text("Error: ${snapshot.error}"));
            } else {
              final dailyData = snapshot.data!;
              return LineChart(
                LineChartData(
                  lineBarsData: [
                    LineChartBarData(
                      spots: dailyData.asMap().entries.map((entry) {
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
                          if (idx < 0 || idx >= dailyData.length) return Container();
                          return Text(dailyData[idx]['label']);
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
                            "${dailyData[idx]['label']}\n₹${spot.y.toStringAsFixed(0)}",
                            const TextStyle(color: Colors.white),
                          );
                        }).toList();
                      },
                    ),
                    touchCallback: (FlTouchEvent event, LineTouchResponse? response) async {
                      // ✅ Only react to tap-up (click), ignore hovers/drags
                      if (event is! FlTapUpEvent) return;
                      if (response == null) return;

                      final spot = response.lineBarSpots?.first;
                      if (spot != null) {
                        final idx = spot.x.toInt();
                        if (idx >= 0 && idx < dailyData.length) {
                          final date = dailyData[idx]['date'];
                          final breakdown = await fetchCategoryBreakdown(date);

                          showModalBottomSheet(
                            context: context,
                            builder: (_) => ListView(
                              children: breakdown.entries.map((e) {
                                return ListTile(
                                  title: Text(e.key),
                                  trailing: Text("₹${e.value.toStringAsFixed(0)}"),
                                );
                              }).toList(),
                            ),
                          );
                        }
                      }
                    },
                  ),
                ),
              );
            }
          },
        ),
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

    // Initialize the formatted time once the widget is built
    _updateDateTime();

    // Update every second
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (mounted) {
        setState(() => _updateDateTime());
      }
    });
  }

  void _updateDateTime() {
    final appState = context.read<MyAppState>();
    DateTime now = appState.currentDate;

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
class DatePickerButton extends StatelessWidget {
  const DatePickerButton({Key? key}) : super(key: key);

  void _showCalendar(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) {
        return Dialog(
          child: CalendarPopup(),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return IconButton(
      onPressed: () => _showCalendar(context),
      icon: Icon(Icons.calendar_today),
      tooltip: 'Select Date', // optional, shows on long press for accessibility
    );
  }
}

class CalendarPopup extends StatefulWidget {
  @override
  _CalendarPopupState createState() => _CalendarPopupState();
}

class _CalendarPopupState extends State<CalendarPopup> {
  CalendarFormat _calendarFormat = CalendarFormat.month;
  DateTime _focusedDay = DateTime.now();
  DateTime? _selectedDay;

  @override
  Widget build(BuildContext context) {
    final appState = context.read<MyAppState>();

    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          TableCalendar(
            firstDay: DateTime(2000),
            lastDay: DateTime.now(),
            focusedDay: _focusedDay,
            calendarFormat: _calendarFormat,
            // 3. This callback will now work correctly
            onFormatChanged: (format) {
              setState(() {
                _calendarFormat = format;
              });
            },
            selectedDayPredicate: (day) => isSameDay(_selectedDay, day),
            onDaySelected: (selectedDay, focusedDay) {
              setState(() {
                _selectedDay = selectedDay;
                _focusedDay = focusedDay;
              });

              // Update app state and fetch notes
              appState.setCurrentDate(selectedDay);
              final formattedDate = DateFormat('yyyy-MM-dd').format(selectedDay);
              appState.fetchNotes(date: formattedDate);

              Navigator.of(context).pop(); // close the popup
            },
            calendarStyle: CalendarStyle(
              todayDecoration: BoxDecoration(
                color: Colors.blue,
                shape: BoxShape.circle,
              ),
              selectedDecoration: BoxDecoration(
                color: Colors.orange,
                shape: BoxShape.circle,
              ),
            ),
          ),
          SizedBox(height: 12),
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: Text("Cancel"),
          ),
        ],
      ),
    );
  }
}
class SplashScreen extends StatefulWidget {
  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();

    _controller = AnimationController(
      duration: const Duration(seconds: 1), // Animation duration
      vsync: this,
    );

    _animation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeIn,
    );

    _controller.forward(); // Start the animation

    // Navigate after a longer delay
    Future.delayed(Duration(seconds: 3), () {
      if (mounted) {
        Navigator.of(context).pushReplacement(
          MaterialPageRoute(builder: (_) => MyHomePage()),
        );
      }
    });
  }

  @override
  void dispose() {
    _controller.dispose(); // Don't forget to dispose the controller
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // Change this line from Colors.blue
      backgroundColor: Theme.of(context).colorScheme.primaryContainer,
      body: Center(
        child: FadeTransition(
          opacity: _animation,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // To match the theme better, you might want the icon to use the primary color
              Icon(
                Icons.account_balance_wallet,
                size: 80,
                color: Theme.of(context).colorScheme.primary, // Changed from Colors.white
              ),
              SizedBox(height: 16),
              Text(
                "FinanceKoi",
                style: GoogleFonts.lato(
                  fontSize: 28,
                  // And the text to use a color that's readable on the new background
                  color: Theme.of(context).colorScheme.onPrimaryContainer, // Changed from Colors.white
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// Settings Page
class SettingsPage extends StatelessWidget {
  const SettingsPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).colorScheme.primaryContainer,
      body: Center(
        child: Text(
          'Settings Page',
          style: GoogleFonts.lato(fontSize: 24),
        ),
      ),
    );
  }
}
const categories = [
  "Education",
  "Food",
  "Healthcare",
  "Housing",
  "Others",
  "Transport"
];

class ReceiptItem {
  String name;
  int quantity;
  double price;
  String category;

  ReceiptItem({
    required this.name,
    required this.quantity,
    required this.price,
    this.category = "Others",
  });

  factory ReceiptItem.fromJson(Map<String, dynamic> json) {
    return ReceiptItem(
      name: json['name'],
      quantity: json['quantity'],
      price: (json['price'] as num).toDouble(),
      category: json['category'] ?? "Others",
    );
  }

  Map<String, dynamic> toJson() {
    return {
      "name": name,
      "quantity": quantity,
      "price": price,
      "category": category,
    };
  }
}
class ReceiptPreviewScreen extends StatefulWidget {
  final List<ReceiptItem> items;
  final double total;

  const ReceiptPreviewScreen({
    super.key,
    required this.items,
    required this.total,
  });

  @override
  State<ReceiptPreviewScreen> createState() => _ReceiptPreviewScreenState();
}

class _ReceiptPreviewScreenState extends State<ReceiptPreviewScreen> {
  late List<ReceiptItem> editableItems;
  double total = 0.0;

  @override
  void initState() {
    super.initState();
    editableItems = List.from(widget.items); // copy
    _recalculateTotal();
  }

  void _recalculateTotal() {
    total = editableItems.fold(
      0.0,
          (sum, item) => sum + (item.quantity * item.price),
    );
    setState(() {}); // update UI
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Receipt Preview")),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              itemCount: editableItems.length,
              itemBuilder: (context, index) {
                final item = editableItems[index];
                return Card(
                  key: ObjectKey(item), // Added key here
                  child: Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            const Text("Item details",
                                style: TextStyle(fontWeight: FontWeight.bold)),
                            IconButton(
                              icon: const Icon(Icons.delete, color: Colors.red),
                              onPressed: () {
                                setState(() {
                                  editableItems.removeAt(index);
                                  _recalculateTotal();
                                });
                              },
                            ),
                          ],
                        ),
                        TextFormField(
                          initialValue: item.name,
                          onChanged: (val) => item.name = val,
                          decoration:
                          const InputDecoration(labelText: "Item"),
                        ),
                        Row(
                          children: [
                            Flexible(
                              child: TextFormField(
                                initialValue: item.quantity.toString(),
                                keyboardType: TextInputType.number,
                                onChanged: (val) {
                                  item.quantity =
                                      int.tryParse(val) ?? item.quantity;
                                  _recalculateTotal();
                                },
                                decoration:
                                const InputDecoration(labelText: "Qty"),
                              ),
                            ),
                            const SizedBox(width: 10),
                            Flexible(
                              child: TextFormField(
                                initialValue: item.price.toString(),
                                keyboardType: TextInputType.number,
                                onChanged: (val) {
                                  item.price =
                                      double.tryParse(val) ?? item.price;
                                  _recalculateTotal();
                                },
                                decoration:
                                const InputDecoration(labelText: "Price"),
                              ),
                            ),
                          ],
                        ),
                        DropdownButtonFormField<String>(
                          value: item.category,
                          items: categories
                              .map((cat) => DropdownMenuItem(
                            value: cat,
                            child: Text(cat),
                          ))
                              .toList(),
                          onChanged: (val) {
                            if (val != null) {
                              setState(() {
                                item.category = val;
                              });
                            }
                          },
                          decoration: const InputDecoration(
                            labelText: "Category",
                            border: OutlineInputBorder(),
                          ),
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
          Text(
            "Total: ${total.toStringAsFixed(2)}",
            style: const TextStyle(fontSize: 18),
          ),
          ElevatedButton(
            onPressed: () async {
              final payload = {
                "items": editableItems.map((e) => e.toJson()).toList(),
              };

              final url = Uri.parse("http://127.0.0.1:8000/receipt/confirm");

              try {
                final response = await http.post(
                  url,
                  headers: {"Content-Type": "application/json"},
                  body: jsonEncode(payload),
                );

                if (response.statusCode == 200) {
                  final resData = jsonDecode(response.body);
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                      content: Text(
                        "Saved! Total: ${resData['entries'].fold(0, (sum, e) => sum + e['total'])}",
                      ),
                    ),
                  );
                } else {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                        content:
                        Text("Failed: ${response.statusCode}")),
                  );
                }
              } catch (e) {
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text("Error: $e")),
                );
              }
            },
            child: const Text("Confirm & Save"),
          ),
        ],
      ),
    );
  }
}
