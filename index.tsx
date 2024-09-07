import React from 'react';
import { Text, View, StyleSheet } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import IntroScreen from './IntroScreen';


export default function App() {
  return (
    <View style={styles.container}>
      <Text style={styles.text}>Hi!</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#FFF',
  },
  text: {
    fontSize: 88,
    color: '#000',
  },
});
