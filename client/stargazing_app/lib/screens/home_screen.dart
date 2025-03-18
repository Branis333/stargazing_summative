// lib/screens/home_screen.dart
import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'package:flutter_datetime_picker_plus/flutter_datetime_picker_plus.dart';
import 'package:intl/intl.dart';
import '../services/api_service.dart';
import '../models/prediction_result.dart';
import 'result_screen.dart';

class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final apiService = StargazingApiService();
  
  double? latitude;
  double? longitude;
  DateTime selectedDateTime = DateTime.now();
  bool isLoading = false;
  String locationName = "Select Location";

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Stargazing Predictor'),
        backgroundColor: Colors.indigo.shade800,
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Colors.indigo.shade900, Colors.black],
          ),
        ),
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // App logo or image
              Image.asset(
                'assets/Skywacth.png', 
                height: 120,
              ),
              SizedBox(height: 24),
              
              // Location card
              Card(
                color: Colors.indigo.shade700,
                child: ListTile(
                  leading: Icon(Icons.location_on, color: Colors.white),
                  title: Text(
                    locationName,
                    style: TextStyle(color: Colors.white),
                  ),
                  trailing: Icon(Icons.arrow_forward_ios, color: Colors.white),
                  onTap: () async {
                    await _getCurrentLocation();
                  },
                ),
              ),
              SizedBox(height: 16),
              
              // Date & Time selector
              Card(
                color: Colors.indigo.shade700,
                child: ListTile(
                  leading: Icon(Icons.access_time, color: Colors.white),
                  title: Text(
                    DateFormat('MMM dd, yyyy - HH:mm').format(selectedDateTime),
                    style: TextStyle(color: Colors.white),
                  ),
                  trailing: Icon(Icons.arrow_forward_ios, color: Colors.white),
                  onTap: () {
                    DatePicker.showDateTimePicker(
                      context,
                      showTitleActions: true,
                      onConfirm: (date) {
                        setState(() {
                          selectedDateTime = date;
                        });
                      },
                      currentTime: selectedDateTime,
                    );
                  },
                ),
              ),
              
              Spacer(),
              
              // Predict button
              ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.amber,
                  foregroundColor: Colors.black,
                  padding: EdgeInsets.symmetric(vertical: 16),
                  textStyle: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                onPressed: (latitude != null && longitude != null) 
                  ? () async {
                      await _predictStargazing();
                    }
                  : null,
                child: isLoading 
                  ? CircularProgressIndicator(color: Colors.black)
                  : Text('PREDICT STARGAZING CONDITIONS'),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _getCurrentLocation() async {
    setState(() {
      isLoading = true;
    });
    
    try {
      // Check location permissions
      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
        if (permission == LocationPermission.denied) {
          throw Exception('Location permission denied');
        }
      }
      
      // Get current position
      Position position = await Geolocator.getCurrentPosition();
      setState(() {
        latitude = position.latitude;
        longitude = position.longitude;
        locationName = "Current Location (${position.latitude.toStringAsFixed(4)}, ${position.longitude.toStringAsFixed(4)})";
        isLoading = false;
      });
    } catch (e) {
      setState(() {
        isLoading = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error getting location: $e')),
      );
    }
  }

  Future<void> _predictStargazing() async {
    if (latitude == null || longitude == null) return;
    
    setState(() {
      isLoading = true;
    });
    
    try {
      final result = await apiService.predictStargazingQuality(
        latitude!,
        longitude!,
        selectedDateTime,
      );
      
      setState(() {
        isLoading = false;
      });
      
      // Navigate to results screen
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => ResultScreen(
            result: PredictionResult.fromJson(result),
            latitude: latitude!,
            longitude: longitude!,
          ),
        ),
      );
    } catch (e) {
      setState(() {
        isLoading = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e')),
      );
    }
  }
}