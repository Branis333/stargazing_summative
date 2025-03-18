// lib/main.dart
import 'package:flutter/material.dart';
import 'screens/home_screen.dart';

void main() {
  runApp(StargazingApp());
}

class StargazingApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Stargazing Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.indigo,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        fontFamily: 'Montserrat',
      ),
      darkTheme: ThemeData.dark().copyWith(
        primaryColor: Colors.indigo,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      themeMode: ThemeMode.dark,
      home: HomeScreen(),
    );
  }
}