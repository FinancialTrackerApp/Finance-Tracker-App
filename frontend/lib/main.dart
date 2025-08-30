import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(MyApp()); // Entry point
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

// Application state: stores the input text and totals
class MyAppState extends ChangeNotifier {
  String userInput = ""; // Current input text
  Map<String, dynamic> totals = {}; // Updated totals from backend
  bool isLoading = false; // Loading indicator

  void updateInput(String newText) {
    userInput = newText;
    notifyListeners();
  }
  // Call Flask backend predict API
  Future<void> predict() async {
    if (userInput.isEmpty) return;
    isLoading = true;
    notifyListeners();
    try {
      final url = Uri.parse("http://127.0.0.1:5000/predict"); // Flask endpoint
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
}
// Main home page with NavigationRail
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
        page = GeneratorPage(); // Input & Predict page
        break;
      default:
        throw UnimplementedError('no widget for $selectedIndex');
    }
    return LayoutBuilder(
        builder: (context, constraints) {
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
        }
    );
  }
}
// Page with input, Predict button, and totals display
class GeneratorPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    var appState = context.watch<MyAppState>();
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // Input TextField
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: TextField(
              onChanged: (text) {
                appState.updateInput(text);
              },
              decoration: InputDecoration(
                border: OutlineInputBorder(),
                labelText: "Enter expense text",
              ),
            ),
          ),
          SizedBox(height: 20),
          // Predict button
          ElevatedButton.icon(
            onPressed: () {
              appState.predict(); // Send input to Flask backend
            },
            icon: Icon(Icons.arrow_right),
            label: Text('Predict'),
          ),
          SizedBox(height: 20),
          // Totals display box
          Container(
            padding: EdgeInsets.all(16),
            margin: EdgeInsets.symmetric(horizontal: 20),
            decoration: BoxDecoration(
              border: Border.all(color: Colors.black54),
              borderRadius: BorderRadius.circular(8),
              color: Colors.white,
            ),
            child: appState.isLoading
                ? Center(child: CircularProgressIndicator())
                : appState.totals.isEmpty
                ? Text("Totals will appear here")
                : Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: appState.totals.entries
                  .map((e) => Text("${e.key}: ₹${e.value}"))
                  .toList(),
            ),
          ),
        ],
      ),
    );
  }
}
