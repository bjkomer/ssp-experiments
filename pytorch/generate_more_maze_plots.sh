for encoding in ssp random 2d one-hot trig random-proj random-trig learned
do
  case "$encoding" in
          ssp )
                  encstr=ssp ;;
          random )
                  encstr=rand ;;
          2d )
                  encstr=2d ;;
          one-hot )
                  encstr=oh ;;
          trig )
                  encstr=trig ;;
          random-proj )
                  encstr=randproj ;;
          random-trig )
                  encstr=rand_trig ;;
          learned )
                  encstr=learned ;;
  esac
  python test_multi_maze_solve_function.py --folder figure_output_50mazes --spatial-encoding $encoding --maze-id-type one-hot --load-saved-model multi_maze_solve_function/encodings/50mazes/${encstr}_loc_oh_id/*/model.pt
done
