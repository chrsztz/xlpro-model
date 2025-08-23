% Test LilyPond content for fingering injection
\version "2.14.0"

trackBchannelB = \relative c-1 {
  \voiceOne
  c'4-2 d-3 e-4 f-5 
  | % 2
  g-1 a-2 b-3 c-4 
  | % 3
}

trackBchannelBvoiceB = \relative c-5 {
  \voiceTwo
  c1-1 
  | % 2
  g' 
  | % 3
}

% 和弦测试
chordTest = {
  <c-2 e-3 g-5>4 <d-4 f-5 a-1> <e-1 g-2 b-2> <f-3 a-4 c-5>
}

\score {
  <<
    \context Staff=trackB \trackBchannelB
    \context Staff=trackC \trackBchannelBvoiceB
  >>
  \layout {}
  \midi {}
}