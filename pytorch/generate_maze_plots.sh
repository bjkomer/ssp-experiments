for encoding in ssp random 2d one-hot trig random-proj
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
  esac
  python test_multi_maze_solve_function.py --spatial-encoding $encoding --maze-id-type one-hot --load-saved-model multi_maze_solve_function/encodings/maze/${encstr}_loc_oh_id/*/model.pt
done
