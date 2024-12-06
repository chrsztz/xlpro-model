import React, { useEffect } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Animated, { useSharedValue, useAnimatedStyle, withTiming } from 'react-native-reanimated';

const IntroScreen: React.FC = () => {
  // 使用 shared value 来控制动画的透明度
  const opacity = useSharedValue(0);

  // 定义动画样式
  const animatedStyle = useAnimatedStyle(() => {
    return {
      opacity: withTiming(opacity.value, { duration: 2000 }), // 动画持续时间2秒
    };
  });

  // 在组件加载时触发动画
  useEffect(() => {
    opacity.value = 1; // 触发动画，使文字从透明变为不透明
  }, []);

  return (
    <View style={styles.container}>
      <Animated.View style={[styles.textContainer, animatedStyle]}>
        <Text style={styles.text}>Welcome to the App!</Text>
        <Text style={styles.text}>This is your introduction screen.</Text>
      </Animated.View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#1e56A0', // 白色背景
  },
  textContainer: {
    paddingHorizontal: 20,
  },
  text: {
    fontSize: 24,
    color: '#F6F6F6', // 黑色文字
    marginVertical: 10,
    textAlign: 'center',
  },
});

export default IntroScreen;
