% Lily was here -- automatically converted by midi2ly from test_midi2ly_output/test.mid
\version "2.14.0"

\layout {
  \context {
    \Voice
    \remove Note_heads_engraver
    \consists Completion_heads_engraver
    \remove Rest_engraver
    \consists Completion_rest_engraver
  }
}

trackAchannelA = {
  
  \tempo 4 = 120 
  
  \time 4/4 
  \skip 4*1/220 
}

trackA = <<
  \context Voice = voiceA \trackAchannelA
>>


trackBchannelA = {
  
  \set Staff.instrumentName = "Piano"
  \skip 4*1761/220 
}

trackBchannelB = \relative c {
  \voiceOne
  c'4 d e f 
  | % 2
  g a b c 
  | % 3
  
}

trackBchannelBvoiceB = \relative c {
  \voiceTwo
  c1 
  | % 2
  g' 
  | % 3
  
}

trackB = <<
  \context Voice = voiceA \trackBchannelA
  \context Voice = voiceB \trackBchannelB
  \context Voice = voiceC \trackBchannelBvoiceB
>>


\score {
  <<
    \context Staff=trackB \trackA
    \context Staff=trackB \trackB
  >>
  \layout {}
  \midi {}
}
