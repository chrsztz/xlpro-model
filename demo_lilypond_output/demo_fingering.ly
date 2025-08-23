\version "2.22.0"

\header {
  title = "钢琴指法演示"
  subtitle = "MIDI → AI预测 → LilyPond → PDF"
  composer = "XLPro Fingering System"
}

\score {
  \new PianoStaff <<
    \new Staff = "right" {
      \clef treble
      \time 4/4
      \tempo 4 = 120
      c'4-1 d'4-2 e'4-3 f'4-4 g'4-5
    }
    \new Staff = "left" {
      \clef bass
      \time 4/4
      c2-5 g2-1
    }
  >>
  \layout {
    \context {
      \Score
      \override SpacingSpanner.spacing-increment = #1.5
    }
  }
  \midi {}
}